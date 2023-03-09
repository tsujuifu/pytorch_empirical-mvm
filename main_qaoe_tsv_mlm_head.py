
from utils.lib import *
from dataset import get_tsv_dls
from main_qaoe_tsv import Dataset_QAOE_TSV
from main_qaoe_lsmdc_fib import Agent_QAOE_LSMDC
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import NoOp, is_main_process, all_gather, get_rank, get_world_size, iter_tqdm
from main_qaoe_lsmdc_fib import VIOLET_QAOE_LSMDC as VIOLET_QAOE_MLM_Head

from tqdm import tqdm

class Dataset_QAOE_MLM_Head(Dataset_QAOE_TSV):
    def __init__(self, args, img_tsv_path, txt, id2lineidx, split, tokzr=None):
        super().__init__(args, img_tsv_path, txt, id2lineidx, split, tokzr=tokzr)
        
        total_examples = len(self.txt)
        invalid_examples = 0
        for item in self.txt:
            ans = item['answer_text']
            ans_id = self.tokzr.convert_tokens_to_ids([ans])[0]
            if ans_id==self.unk_token_id: invalid_examples += 1
        LOGGER.info(f"Split {split}, Invalid examples: {invalid_examples} "
                    f"/ Total examples: {total_examples}, "
                    f"upper-bound: {(1 - invalid_examples/total_examples)*100:.2f}%")

    def append_mask(self, tokens, padding_len):
        tokens = [self.tokzr.cls_token]+tokens+self.tokzr.tokenize(f"answer: ")+[self.tokzr.mask_token, self.tokzr.sep_token]+[self.tokzr.pad_token]*(padding_len)
        return tokens

    def prepend_mask(self, tokens, padding_len):
        tokens = [self.tokzr.mask_token]+[self.tokzr.cls_token]+tokens+[self.tokzr.sep_token]+[self.tokzr.pad_token]*(padding_len)
        return tokens

    def replace_cls(self, tokens, padding_len):
        tokens = [self.tokzr.mask_token]+tokens+[self.tokzr.sep_token]+[self.tokzr.pad_token]*(padding_len)
        return tokens

    def insert_mask(self, tokens, padding_len):
        tokens = [self.tokzr.cls_token]+tokens+[self.tokzr.sep_token]+[self.tokzr.pad_token]*(padding_len)
        if len(tokens)<10: tokens += [self.tokzr.mask_token]
        else: tokens = tokens[:10]+[self.tokzr.mask_token]+tokens[10:]
        return tokens

    def str2txt(self, s):
        tokens = self.tokzr.tokenize(s)
        tokens = tokens[:self.args.size_txt-1]
        padding_len = self.args.size_txt-len(tokens)
        if self.args.mask_pos=="append": tokens = self.append_mask(tokens, padding_len)
        elif self.args.mask_pos=="prepend": tokens = self.prepend_mask(tokens, padding_len)
        elif self.args.mask_pos=="insert": tokens = self.insert_mask(tokens, padding_len)
        elif self.args.mask_pos=="replace": tokens = self.replace_cls(tokens, padding_len)
        txt = self.tokzr.convert_tokens_to_ids(tokens)
        
        mask = [1 if w!=self.pad_token_id else 0 for w in txt]
        mask = T.LongTensor(mask)
        txt = T.LongTensor(txt)
        return txt, mask

    @property
    def prompt_text(self):
        return "answer the question about the video."

    def __getitem__(self, idx):
        item = self.txt[idx]
        video_id = item['video']
        if video_id in self.id2lineidx:
            lineidx = self.id2lineidx[video_id]
            b = self.seek_img_tsv(lineidx)[2:]
            img = self.get_img_or_video(b)
        else:
            print(f"video missing: {video_id}")
            img = T.zeros((self.args.size_frame, 3, self.args.size_img, self.args.size_img))

        q = item['question']
        txt, mask = self.str2txt(q)

        if self.args.size_vocab>0: ans_id = item['answer']
        else:
            ans = item['answer_text']
            ans_id = self.tokzr.convert_tokens_to_ids([ans])[0]
            if ans_id==self.unk_token_id: ans_id = -1
        if video_id not in self.id2lineidx:
            print(f"video {video_id} not found")
            ans_id = -1
        mask_ans = T.ones(txt.shape).long()*-1
        mask_ans[txt==self.mask_token_id] = ans_id
        return img, txt, mask, mask_ans

    def collate_batch(self, inputs):
        img, txt, mask, mask_ans = map(list, unzip(inputs))

        all_imgs = T.stack(img, dim=0)
        all_mask_ans = T.stack(mask_ans, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)

        batch = {"img": all_imgs, "txt": all_txts, "mask": all_masks, "mask_ans": all_mask_ans}
        return batch

class Agent_QAOE_MLM_Head(Agent_QAOE_LSMDC):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        if args.freeze_violet: self.model.freeze()

    def go_dl(self, ep, dl, is_train):
        if is_train: self.model.train()
        else: self.model.eval()
        ret = defaultdict(list)
        idx = 0
        for idx, batch in iter_tqdm(enumerate(dl)):
            if is_train: self.global_step += 1
            if (idx%self.args.logging_steps)==0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))
            if self.args.enable_prompt: batch["prompt"] = dl.dataset.get_prompt()
            elif self.args.enable_task_token: batch["task_name"] = "oe"
            batch = self.prepare_batch(batch)
            r = self.step(batch, is_train)
            if is_train: self.log_dict_to_wandb({f"train_{k}": l for k, l in r.items()})
            ret = {k: ret[k]+l if isinstance(l, list) else ret[k]+[l] for k, l in r.items()}

        if (idx%self.args.logging_steps)!=0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))

        gathered_ret = defaultdict(list)
        for ret_per_rank in all_gather(ret):
            for k in ret_per_rank: gathered_ret[k].extend(ret_per_rank[k])
        ret_all = {k: float(np.average(gathered_ret[k])) for k in ret}
        return ret_all

if __name__=='__main__':
    args = get_args()
    args.size_vocab = -1
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    dl_tr, dl_vl, dl_ts = get_tsv_dls(args, Dataset_QAOE_MLM_Head, tokzr=tokzr)
    print(len(dl_tr), len(dl_vl), len(dl_ts))
    
    if args.size_epoch==0: args.max_iter = 1
    else: args.max_iter = len(dl_tr)*args.size_epoch
    args.actual_size_test = len(dl_ts.dataset)

    model = VIOLET_QAOE_MLM_Head(args, tokzr=tokzr)
    model.load_ckpt(args.path_ckpt)
    if args.reinit_head: model.reinit_head()
    model.cuda()

    if args.distributed: LOGGER.info(f"n_gpu: {args.num_gpus}, rank: {get_rank()},"
                                     f" world_size: {get_world_size()}")

    args.path_output = '%s/_%s_%s'%(args.path_output, args.task, datetime.now().strftime('%Y%m%d%H%M%S'))
    agent = Agent_QAOE_MLM_Head(args, model)
    if args.distributed: agent.prepare_dist_model()
    agent.save_training_meta()
    if is_main_process(): add_log_to_file('%s/stdout.txt' % (args.path_output))
    else: LOGGER = NoOp()
    LOGGER.info("Saved training meta infomation...")

    if os.path.exists(args.path_ckpt):
        LOGGER.info("Zero-shot Evaluation")
        if len(dl_vl):
            ac_vl = agent.go_dl(0, dl_vl, False)
            LOGGER.info(f'ZS (val): {ac_vl["ac_1"]*100:.2f}, {ac_vl["ac_5"]*100:.2f}')
            print('ZS (val):', ac_vl["ac_1"], ac_vl["ac_5"])
        if len(dl_ts):
            ac_ts = agent.go_dl(0, dl_ts, False)
            LOGGER.info(f'ZS (test): {ac_ts["ac_1"]*100:.2f}, {ac_ts["ac_5"]*100:.2f}')
            print('ZS (test):', ac_ts["ac_1"], ac_ts["ac_5"])
            if (hasattr(args, "size_test") and args.size_test!=args.actual_size_test):
                adjusted_ac_ts_1 = ac_ts['ac_1']*args.actual_size_test/args.size_test*100
                adjusted_ac_ts_5 = ac_ts['ac_5']*args.actual_size_test/args.size_test*100
                LOGGER.info(f'ZS (test, adjusted): {adjusted_ac_ts_1:.2f}'
                            f', {adjusted_ac_ts_5:.2f}')
                print('ZS (test, adjusted):', adjusted_ac_ts_1, adjusted_ac_ts_5)
    else: LOGGER.info("No pre-trained weight, skip zero-shot Evaluation")
        
    if args.size_epoch:
        agent.setup_wandb()
        LOGGER.info("Start training....")
        for e in iter_tqdm(range(args.size_epoch)):
            ls_tr = agent.go_dl(e+1, dl_tr, True)
            LOGGER.info(f'Ep {e}, Loss (train): {ls_tr["ls"]*100:.4e}')
            
            if len(dl_vl):
                ac_vl = agent.go_dl(e+1, dl_vl, False)
                for k in ac_vl:
                    agent.log[f'{k}_vl'].append(ac_vl[k])
                    agent.log_dict_to_wandb({"{k}_vl": ac_vl[k]})
                LOGGER.info(f'Ep {e}, Acc (val): {ac_vl["ac_1"]*100:.2f}, '
                            f'{ac_vl["ac_5"]*100:.2f}')
            if len(dl_ts):
                ac_ts = agent.go_dl(e+1, dl_ts, False)
                LOGGER.info(f'Ep {e}, Acc (test): {ac_ts["ac_1"]*100:.2f}, '
                            f'{ac_ts["ac_5"]*100:.2f}')
                if (hasattr(args, "size_test") and args.size_test!=args.actual_size_test):
                    adjusted_ac_ts_1 = ac_ts['ac_1']*args.actual_size_test/args.size_test
                    adjusted_ac_ts_5 = ac_ts['ac_5']*args.actual_size_test/args.size_test
                    agent.log['ac_1_ts'].append(adjusted_ac_ts_1)
                    agent.log['ac_5_ts'].append(adjusted_ac_ts_5)
                    LOGGER.info(f'Ep {e}, Acc (test, adjusted): {adjusted_ac_ts_1*100:.2f}'
                                f', {adjusted_ac_ts_5*100:.2f}')
                else:
                    for k in ac_ts:
                        agent.log_dict_to_wandb({"{k}_ts": ac_ts[k]})
                        agent.log[f'{k}_ts'].append(ac_ts[k])
            agent.save_model(e+1)
        best_vl, best_ts = agent.best_epoch()
        LOGGER.info(f'Best val @ ep {best_vl[0]+1}, {best_vl[1]*100:.2f}')
        LOGGER.info(f'Best test @ ep {best_ts[0]+1}, {best_ts[1]*100:.2f}'
                    f' (adjusted)')
        