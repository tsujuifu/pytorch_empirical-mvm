
from utils.lib import *
from main_qamc import VIOLET_QAMC, Dataset_QAMC, Agent_QAMC
from dataset import get_tsv_dls
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import NoOp, is_main_process, all_gather, get_rank, get_world_size, iter_tqdm

class Dataset_QAMC_TSV(Dataset_QAMC):
    def __init__(self, args, img_tsv_path, txt, id2lineidx, split, tokzr=None):
        super(Dataset_QAMC, self).__init__(args, split, size_frame=args.size_frame, tokzr=tokzr)
        
        self.txt = txt[split]
        self.img_tsv_path = img_tsv_path
        self.id2lineidx = id2lineidx
        if args.data_ratio!=1: self.get_partial_data()

    def __getitem__(self, idx):
        item = self.txt[idx]
        video_id = item['video']
        lineidx = self.id2lineidx[video_id]
        b = self.seek_img_tsv(lineidx)[2:]
        img = self.get_img_or_video(b)
        q = item['question']

        txt, mask = [], []
        for i in range(self.args.size_option):
            if len(q): option = q+' '+item[f'option_{i}']
            else: option = item[f'option_{i}']
            t, m = self.str2txt(option)
            txt.append(t), mask.append(m)
        txt = T.stack(txt)
        mask = T.stack(mask)

        return img, txt, mask, item['answer']

class Agent_QAMC_TSV(Agent_QAMC):
    def __init__(self, args, model):
        super().__init__(args, model)

    def go_dl(self, ep, dl, is_train):
        if is_train: self.model.train()
        else: self.model.eval()
        ret = []
        idx = 0
        for idx, batch in iter_tqdm(enumerate(dl)):
            if is_train: self.global_step += 1
            if (idx%self.args.logging_steps)==0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))
            batch = self.prepare_batch(batch)
            img, txt, mask, ans = [batch[key] for key in ["img", "txt", "mask", "ans"]]
            curr_ret = self.step(img, txt, mask, ans, is_train)
            if is_train: self.log_dict_to_wandb({"train_ls": curr_ret})
            if isinstance(curr_ret, list): ret.extend(curr_ret)
            else: ret.append(curr_ret)

        if (idx%self.args.logging_steps)!=0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))

        gathered_ret = []
        for ret_per_rank in all_gather(ret): gathered_ret.extend(ret_per_rank)
        ret = float(np.average(gathered_ret))

        return ret

if __name__=='__main__':
    args = get_args()
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    
    dl_tr, dl_vl, dl_ts = get_tsv_dls(args, Dataset_QAMC_TSV, tokzr=tokzr)
    print(len(dl_tr.dataset), len(dl_vl.dataset), len(dl_ts.dataset))
    
    if args.size_epoch==0 or len(dl_tr)==0: args.max_iter = 1
    else: args.max_iter = len(dl_tr)*args.size_epoch
    args.actual_size_test = len(dl_ts.dataset)

    model = VIOLET_QAMC(args, tokzr=tokzr)
    model.load_ckpt(args.path_ckpt)
    if args.reinit_head: model.reinit_head()
    model.cuda()
    
    if args.distributed: LOGGER.info(f"n_gpu: {args.num_gpus}, rank: {get_rank()},"
                                     f" world_size: {get_world_size()}")

    args.path_output = '%s/_%s_%s'%(args.path_output, args.task, datetime.now().strftime('%Y%m%d%H%M%S'))
    agent = Agent_QAMC_TSV(args, model)
    if args.distributed: agent.prepare_dist_model()
    agent.save_training_meta()
    if is_main_process(): add_log_to_file('%s/stdout.txt' % (args.path_output))
    else: LOGGER = NoOp()
    LOGGER.info("Saved training meta infomation, start training ...")

    if os.path.exists(args.path_ckpt):
        if len(dl_vl): ac_vl = agent.go_dl(0, dl_vl, False)
        else: ac_vl = -0.01
        if len(dl_ts): ac_ts = agent.go_dl(0, dl_ts, False)
        else: ac_ts = -0.01
        LOGGER.info('ZS: %.2f %.2f'%(ac_vl*100, ac_ts*100))
        print('ZS: %.2f %.2f' % (ac_vl*100, ac_ts*100))
    else: LOGGER.info("No pre-trained weight, skip zero-shot Evaluation")

    if args.size_epoch:
        agent.setup_wandb()
        LOGGER.info("Start training....")

        for e in iter_tqdm(range(args.size_epoch)):
            ls_tr = agent.go_dl(e+1, dl_tr, True)

            ac_vl = agent.go_dl(e+1, dl_vl, False)
            ac_ts = agent.go_dl(e+1, dl_ts, False)
            agent.log['ls_tr'].append(ls_tr)
            agent.log['ac_vl'].append(ac_vl)
            agent.log['ac_ts'].append(ac_ts)
            agent.log_dict_to_wandb({"ac_vl": ac_vl})
            agent.log_dict_to_wandb({"ac_ts": ac_ts})
            LOGGER.info('Ep %d: %.6f %.2f %.2f'%(e+1, ls_tr, ac_vl*100, ac_ts*100))
            agent.save_model(e+1)
        best_vl, best_ts = agent.best_epoch()
        LOGGER.info(f'Best val @ ep {best_vl[0]+1}, {best_vl[1]*100:.2f}')
        LOGGER.info(f'Best test @ ep {best_ts[0]+1}, {best_ts[1]*100:.2f}')
        