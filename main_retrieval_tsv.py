
from utils.lib import *
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from dataset import get_tsv_dls
from utils.dist import is_main_process, get_rank, get_world_size, iter_tqdm, NoOp
from main_retrieval import Dataset_Retrieval, VIOLET_Retrieval, Agent_Retrieval

class Dataset_Retrieval_TSV(Dataset_Retrieval):
    def __init__(self, args, img_tsv_path, txt, id2lineidx, split, tokzr):
        super(Dataset_Retrieval, self).__init__(args, split, size_frame=args.size_frame, tokzr=tokzr)
        
        self.txt = txt[split]
        self.img_tsv_path = img_tsv_path
        self.id2lineidx = id2lineidx
        if args.data_ratio!=1: self.get_partial_data()
        self.vid2txt = defaultdict(list)
        for item in self.txt: self.vid2txt[item["video"]].append(item)
        if self.split!="train" and len(self.txt)>len(self.vid2txt):
            first_txt = []
            for vid, list_of_items in self.vid2txt.items(): first_txt.append(list_of_items[0])
            self.txt = first_txt

    def __getitem__(self, idx):
        item = self.txt[idx]
        video_id = item['video']
        lineidx = self.id2lineidx[video_id]
        b = self.seek_img_tsv(lineidx)[2:]
        img = self.get_img_or_video(b)

        raw_txt = item['caption']
        if isinstance(raw_txt, list):
            sent_ids = range(len(raw_txt))
            if self.split=="train":
                size_sent = random.randint(1, len(raw_txt))
                sent_ids = random.sample(sent_ids, size_sent)
            raw_txt = " ".join([raw_txt[i] for i in sent_ids])

        txt, mask = self.str2txt(raw_txt)

        return img, txt, mask, video_id
    
class Agent_Retrieval_TSV(Agent_Retrieval):
    def __init__(self, args, model):
        super().__init__(args, model)

    def go_dl(self, ep, dl, is_train):
        if is_train: self.model.train()
        else: self.model.eval()
        ret = []
        idx = 0
        for idx, batch in iter_tqdm(enumerate(dl)):
            if is_train: self.global_step += 1
            if (self.global_step%self.args.logging_steps)==0 and is_train: LOGGER.info(self.log_memory())
            batch = self.prepare_batch(batch)
            img, txt, mask, vid = [batch[key] for key in ['img', 'txt', 'mask', 'vid']]
            curr_ret = self.step(img, txt, mask, vid, is_train)
            if is_train: self.log_dict_to_wandb({"train_ls": curr_ret})
            ret.append(curr_ret)

        if (idx%self.args.logging_steps)!=0 and is_train: LOGGER.info(self.log_memory())

        ret = float(float(np.average(ret)))
        if self.args.distributed: ret = self.reduce_mean(ret)
        return ret

if __name__=='__main__':
    args = get_args()
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    dl_tr, dl_vl, dl_ts = get_tsv_dls(args, Dataset_Retrieval_TSV, tokzr=tokzr)
    print(len(dl_tr), len(dl_vl), len(dl_ts))
    
    args.max_iter = len(dl_tr)*args.size_epoch

    model = VIOLET_Retrieval(args, tokzr=tokzr)
    model.load_ckpt(args.path_ckpt)
    model.cuda()
    if args.distributed: LOGGER.info(f"n_gpu: {args.num_gpus}, rank: {get_rank()},"
                                     f" world_size: {get_world_size()}")

    args.path_output = '%s/_%s_%s'%(args.path_output, args.task, datetime.now().strftime('%Y%m%d%H%M%S'))
    agent = Agent_Retrieval_TSV(args, model)
    if args.distributed: agent.prepare_dist_model()
    agent.save_training_meta()
    if is_main_process(): add_log_to_file('%s/stdout.txt'%(args.path_output))
    else: LOGGER = NoOp()
        
    agent.setup_wandb()
    LOGGER.info("Saved training meta infomation, start training ...")
    for e in iter_tqdm(range(args.size_epoch)):
        ls_tr = agent.go_dl(e+1, dl_tr, True)
        
        ac_vl = agent.go_dl(e+1, dl_vl, False)
        ac_ts = agent.go_dl(e+1, dl_ts, False)
        agent.log_dict_to_wandb({"ac_vl": ac_vl})
        agent.log_dict_to_wandb({"ac_ts": ac_ts})
        agent.log['ls_tr'].append(ls_tr)
        agent.log['ac_vl'].append(ac_vl)
        agent.log['ac_ts'].append(ac_ts)
        agent.save_model(e+1)
        LOGGER.info('Ep %d: %.6f %.6f %.6f'%(e+1, ls_tr, ac_vl, ac_ts))
    best_vl, best_ts = agent.best_epoch()
    LOGGER.info(f'Best val @ ep {best_vl[0]+1}, {best_vl[1]:.6f}')
    LOGGER.info(f'Best test @ ep {best_ts[0]+1}, {best_ts[1]:.6f}')
    