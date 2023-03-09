
from utils.lib import *
from main_qamc_tsv import VIOLET_QAMC, Dataset_QAMC_TSV, Agent_QAMC
from dataset import get_tsv_dls
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import NoOp, is_main_process, all_gather, get_rank, get_world_size, iter_tqdm
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

class Dataset_QAMC_MLM_Head(Dataset_QAMC_TSV):
    def __init__(self, args, img_tsv_path, txt, id2lineidx, split, tokzr=None):
        super().__init__(args, img_tsv_path, txt, id2lineidx, split, tokzr=tokzr)

    def str2txt(self, s):
        txt, mask = super().str2txt(s)
        txt, mask = self.append_mask_tok2txt(txt, mask)
        return txt, mask

    def __getitem__(self, idx):
        item = self.txt[idx]
        video_id = item['video']
        lineidx = self.id2lineidx[video_id]
        b = self.seek_img_tsv(lineidx)[2:]
        img = self.get_img_or_video(b)
        ans_idx = item['answer']
        q = item['question']

        txt, mask, mask_ans = [], [], []
        for i in range(self.args.size_option):
            if len(q): option = q+' '+item[f'option_{i}']
            else: option = item[f'option_{i}']
            t, m = self.str2txt(option)
            txt.append(t), mask.append(m)
            ma = T.ones(t.shape).long() * -1
            if i==ans_idx: ans_id = self.true_token_id
            else: ans_id = self.false_token_id
            ma[t==self.mask_token_id] = ans_id
            mask_ans.append(ma)

        txt = T.stack(txt)
        mask = T.stack(mask)
        mask_ans = T.stack(mask_ans)

        return img, txt, mask, mask_ans

    @property
    def prompt_text(self):
        return "is the video-text paired, true or false?"

    def collate_batch(self, inputs):
        img, txt, mask, mask_ans = map(list, unzip(inputs))

        all_imgs = T.stack(img, dim=0)
        all_mask_ans = T.stack(mask_ans, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)

        batch = {"img": all_imgs, "txt": all_txts, "mask": all_masks, "mask_ans": all_mask_ans}
        return batch

class VIOLET_QAMC_MLM_Head(VIOLET_QAMC):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)
        
        bert = transformers.AutoModelForMaskedLM.from_pretrained(self.args.tokenizer)
        if isinstance(bert, transformers.RobertaForMaskedLM): self.fc_mtm = bert.lm_head
        else: self.fc_mtm = bert.cls
        del bert
        del self.fc
        self.task_tok2id = {"vtm": 0, "mc": 1, "oe": 2, "cap": 3}
        self.emb_task = T.nn.Parameter(0.02*T.randn(10, self.hidden_size))

    def prepro_pretxt(self, task_or_prompt_txt):
        return T.ones_like(task_or_prompt_txt) * -1

    def forward(self, batch):
        batch = defaultdict(lambda: None, batch)
        img, txt, mask = [batch[key] for key in ["img", "txt", "mask"]]
        ans = batch["mask_ans"]

        (_B, _T, _, _H, _W), (_, _O, _) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt.flatten(0, 1), mask.flatten(0, 1))
        feat_img, mask_img = [feat_img.unsqueeze(1).expand([-1, _O, -1, -1]).flatten(0, 1), 
                              mask_img.unsqueeze(1).expand([-1, _O, -1]).flatten(0, 1)]
        _B, _O, _L = ans.shape
        ans = ans.flatten(0, 1)
        ans, mask_txt, feat_txt = self.prepro_txt_inputs(ans, mask_txt, feat_txt, task_name=batch["task_name"], prompt=batch["prompt"])
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        if self.args.temporal_fusion=="mean": _T = 1
        out = self.fc_mtm(out[:, (1+_h*_w)*_T:])
        ans = ans.view(_B, _O, -1)
        return out, ans

class Agent_QAMC_MLM_Head(Agent_QAMC):
    def __init__(self, args, model):
        super().__init__(args, model)

    def step(self, batch, is_train):
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step(batch)
            out, ans = out
        if is_train:
            ans = ans.flatten(0, 1)
            out = out.flatten(0, len(out.shape)-2)
            ans = ans.flatten(0, len(ans.shape)-1)
            ls = self.loss_func(out, ans)
            self.backward_step(ls)
            return ls.item()
        else:
            _B, _O, _L = ans.shape
            p_true = out[:, :, self.true_token_id]
            p_false = out[:, :, self.false_token_id]
            out_mtm = p_true/(p_true+p_false)
            ans_mtm = ans.view(_B*_O, _L)
            assert ans_mtm.shape==out_mtm.shape
            out_mtm = out_mtm[ans_mtm!=-1].view(_B, _O)
            ans_mtm = ans_mtm[ans_mtm!=-1].view(_B, _O)
            out_mtm = T.argmax(out_mtm, dim=-1)
            ans_mtm_idx = (ans_mtm==self.true_token_id).nonzero()[:, 1]
            ac = (out_mtm==ans_mtm_idx).float().tolist()
            return ac

    def go_dl(self, ep, dl, is_train):
        if is_train: self.model.train()
        else: self.model.eval()
        ret = []
        idx = 0
        for idx, batch in iter_tqdm(enumerate(dl)):
            if is_train: self.global_step += 1
            if (idx%self.args.logging_steps)==0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))
            if self.args.enable_prompt: batch["prompt"] = dl.dataset.get_prompt()
            elif self.args.enable_task_token:
                if ep==0: batch["task_name"] = "vtm"
                else: batch["task_name"] = self.args.task_token

            batch = self.prepare_batch(batch)
            curr_ret = self.step(batch=batch, is_train=is_train)
            if is_train: self.log_dict_to_wandb({"train_ls": curr_ret})
            if isinstance(curr_ret, list): ret.extend(curr_ret)
            else: ret.append(curr_ret)

        if (idx%self.args.logging_steps)!=0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))

        gathered_ret = []
        for ret_per_rank in all_gather(ret): gathered_ret.extend(ret_per_rank)
        num_ex = len(gathered_ret)
        ret = float(np.average(gathered_ret))
        
        return ret
    