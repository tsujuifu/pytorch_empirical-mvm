
from utils.lib import *
from dataset import get_dl
from model import VIOLET_Base
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import NoOp, is_main_process, all_gather, get_rank, get_world_size, iter_tqdm
from main_qaoe import Dataset_QAOE, Agent_QAOE

class Dataset_QAOE_LSMDC(Dataset_QAOE):
    def __init__(self, args, img, txt, split, tokzr=None):
        super().__init__(args, img, txt, split, tokzr=tokzr)
        
        total_examples = len(self.txt)
        invalid_examples = 0
        for item in self.txt:
            ans = self.label2ans[item['answer']]
            ans_id = self.tokzr.convert_tokens_to_ids([ans])[0]
            if ans_id==self.unk_token_id: invalid_examples += 1
        LOGGER.info(f"Split {split}, Invalid examples: {invalid_examples} "
                    f"/ Total examples: {total_examples}, "
                    f"upper-bound: {(1 - invalid_examples/total_examples)*100:.2f}%")

    @property
    def prompt_text(self):
        return "fill in the mask to complete the sentence."

    def __getitem__(self, idx):
        item = self.txt[idx]

        img = self.get_img_or_video(self.img[item['video']])
        q = item['question']
        q = q.replace("[MASK]", self.tokzr.mask_token)
        txt, mask = self.str2txt(q)
        if self.args.size_vocab>0: ans_id = item['answer']
        else:
            ans = self.label2ans[item['answer']]
            ans_id = self.tokzr.convert_tokens_to_ids([ans])[0]
            if ans_id==self.unk_token_id: ans_id = -1
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

class VIOLET_QAOE_LSMDC(VIOLET_Base):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)
        
        assert args.size_vocab==-1
        bert = transformers.AutoModelForMaskedLM.from_pretrained(self.args.tokenizer)
        if isinstance(bert, transformers.RobertaForMaskedLM): self.fc_mtm = bert.lm_head
        else: self.fc_mtm = bert.cls
        del bert
        
        self.task_tok2id = {"vtm": 0, "mc": 1, "oe": 2, "cap": 3}
        self.emb_task = T.nn.Parameter(0.02*T.randn(10, self.hidden_size))

    def prepro_pretxt(self, task_or_prompt_txt):
        return T.ones_like(task_or_prompt_txt)*-1

    def forward(self, batch):
        batch = defaultdict(lambda: None, batch)
        img, txt, mask = [batch[key] for key in ["img", "txt", "mask"]]
        ans = batch["mask_ans"]
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        ans, mask_txt, feat_txt = self.prepro_txt_inputs(ans, mask_txt, feat_txt, task_name=batch["task_name"], prompt=batch["prompt"])
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        if self.args.temporal_fusion=="mean": _T = 1
        out = self.fc_mtm(out[:, (1+_h*_w)*_T:])
        return out, ans

class Agent_QAOE_LSMDC(Agent_QAOE):
    def __init__(self, args, model):
        super().__init__(args, model)

    def step(self, batch, is_train):
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step(batch)
            out, ans = out
        if is_train:
            out = out.flatten(0, len(out.shape)-2)
            ans = ans.flatten(0, len(ans.shape)-1)
            ls = self.loss_func(out, ans)
            self.backward_step(ls)
            return {'ls': ls.item()}
        else:
            ac_1 = self.get_top_k_acc(out, ans, k=1)
            ac_5 = self.get_top_k_acc(out, ans, k=5)
            return {'ac_1': ac_1, 'ac_5': ac_5}

    def get_top_k_acc(self, out, ans, k=5):
        _B = out.shape[0]
        if T.any(ans!=-1):
            ans_mtm = ans[ans!=-1].view(-1, 1)
            n_valid_ans = ans_mtm.shape[0]
            out_mtm = out[ans!=-1].view(n_valid_ans, -1)
            out_mtm_v, out_mtm_i = T.topk(out_mtm, k=k, dim=-1)
            ac = (out_mtm_i==ans_mtm).any(dim=-1).float().tolist()
        else:
            print(T.any(ans!=-1, dim=-1))
            ac = []
        if len(ac)<_B: ac += [0.]*(_B-len(ac))
        return ac

    def best_epoch(self):
        if not hasattr(self, "log"): raise NotImplementedError("no log to find the best epoch")
        if "ac_1_vl" not in self.log or "ac_1_ts" not in self.log: raise ValueError("calling best_epoch in pretraining, maybe?")
        val_index = np.argmax(self.log["ac_1_vl"])
        test_index = np.argmax(self.log["ac_1_ts"])
        val_max = self.log["ac_1_vl"][val_index]
        test_max = self.log["ac_1_ts"][test_index]
        return (val_index, val_max), (test_index, test_max)

    def go_dl(self, ep, dl, is_train):
        if is_train: self.model.train()
        else: self.model.eval()
        ret = defaultdict(list)
        idx = 0
        for idx, batch in enumerate(dl):
            if is_train: self.global_step += 1
            if (idx%self.args.logging_steps)==0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))
            if self.args.enable_prompt: batch["prompt"] = dl.dataset.get_prompt()
            elif self.args.enable_task_token: batch["task_name"] = "oe"

            batch = self.prepare_batch(batch)
            r = self.step(batch, is_train)
            ret = {k: ret[k]+l if isinstance(l, list) else ret[k]+[l] for k, l in r.items()}
            if is_train: self.log_dict_to_wandb({f'train_{k}': l for k, l in r.items()})

        if (idx%self.args.logging_steps)!=0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))

        gathered_ret = defaultdict(list)
        for ret_per_rank in all_gather(ret):
            for k in ret_per_rank: gathered_ret[k].extend(ret_per_rank[k])
        ret_all = {k: float(np.average(gathered_ret[k])) for k in ret}
        return ret_all
    