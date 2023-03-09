
from utils.lib import *
from dataset import Dataset_Base, get_dl
from model import VIOLET_Base
from agent import Agent_Base

from utils.misc import humanbytes
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import is_main_process, get_rank, get_world_size, iter_tqdm, all_gather, NoOp

class Dataset_QAMC(Dataset_Base):
    def __init__(self, args, img, txt, split, tokzr=None):
        super().__init__(args, split, size_frame=args.size_frame, tokzr=tokzr)
        
        self.img, self.txt = img, txt[split]
        if args.data_ratio!=1: self.get_partial_data()

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, idx):
        item = self.txt[idx]

        img = self.get_img_or_video(self.img[item['video']])
        q = item['question']

        txt, mask = [], []
        for i in range(self.args.size_option):
            if len(q): option = q+f' {self.tokzr.sep_token} '+item[f'option_{i}']
            else: option = item[f'option_{i}']
            t, m = self.str2txt(option)
            txt.append(t), mask.append(m)
        txt = T.stack(txt)
        mask = T.stack(mask)

        return img, txt, mask, item['answer']

    def collate_batch(self, inputs):
        img, txt, mask, ans = map(list, unzip(inputs))

        all_imgs = T.stack(img, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)
        all_ans = T.LongTensor(ans)

        batch = {"img": all_imgs, "txt": all_txts, "mask": all_masks, "ans": all_ans}
        return batch

class VIOLET_QAMC(VIOLET_Base):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)
        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.1), T.nn.Linear(self.hidden_size, self.hidden_size*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(self.hidden_size*2, 1)])
        if self.args.num_video_tokens!=-1:
            self.num_attention_heads = self.args.num_video_tokens
            self.attention_head_size = int(self.hidden_size/self.num_attention_heads)
            self.all_head_size = self.num_attention_heads*self.attention_head_size
            self.vid_key = T.nn.Linear(in_features=self.hidden_size, out_features=self.all_head_size, bias=False)
            self.vid_query = T.nn.Linear(in_features=self.hidden_size, out_features=self.all_head_size, bias=False)
            self.vid_dropout = T.nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1]+(self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def select_vid_token(self, feat_img, mask_img):
        key_layer = self.transpose_for_scores(self.vid_key(feat_img))
        query_layer = self.transpose_for_scores(self.vid_query(feat_img))

        attention_scores = T.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores/math.sqrt(self.hidden_size)

        ext_mask_img = self.mask_ext(mask_img, mask_img.shape, mask_img.device)
        ext_mask_img = ext_mask_img.to(dtype=feat_img.dtype)
        attention_scores = attention_scores+ext_mask_img
        attention_probs = T.nn.functional.softmax(attention_scores, dim=-1).sum(dim=-2)
        attention_probs = self.vid_dropout(attention_probs)
        attention_probs = T.nn.functional.gumbel_softmax(attention_probs, tau=self.args.gumble_tau, hard=True, dim=-1).sum(dim=1)
        
        context_mask = mask_img*(attention_probs>0)
        return context_mask

    def forward(self, img, txt, mask, ans):
        (_B, _T, _, _H, _W), (_, _O, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt.flatten(0, 1), mask.flatten(0, 1))
        if self.args.num_video_tokens>-1: mask_img = self.select_vid_token(feat_img, mask_img)

        feat_img, mask_img = [feat_img.unsqueeze(1).expand([-1, _O, -1, -1]).flatten(0, 1), 
                              mask_img.unsqueeze(1).expand([-1, _O, -1]).flatten(0, 1)]
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        if self.args.temporal_fusion=="mean": _T = 1
        out = self.fc(out[:, (1+_h*_w)*_T, :]).squeeze(dim=-1).view([_B, _O])

        return out, ans

    def reinit_head(self):
        del self.fc
        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.1), T.nn.Linear(self.hidden_size, self.hidden_size*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(self.hidden_size*2, 1)])

class Agent_QAMC(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        self.log = {'ls_tr': [], 'ac_vl': [], 'ac_ts': []}

    def build_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

        decay_swin_param = [(n, p) for n, p in decay_param_tp if n.startswith("fc.")]
        decay_other_param = [(n, p) for n, p in decay_param_tp if not n.startswith("fc.")]
        print([n for (n, p) in decay_swin_param])

        no_decay_swin_param = [(n, p) for n, p in no_decay_param_tp if n.startswith("fc.")]
        no_decay_other_param = [(n, p) for n, p in no_decay_param_tp if not n.startswith("fc.")]

        weight_decay = self.args.decay
        coef_lr = self.args.vis_backbone_lr_mul
        lr = self.args.lr
        optimizer_grouped_parameters = [{'params': [p for n, p in decay_swin_param], 
                                         'weight_decay': weight_decay, 
                                         'lr': lr * coef_lr}, 
                                        {'params': [p for n, p in decay_other_param], 
                                         'weight_decay': weight_decay}, 
                                        {'params': [p for n, p in no_decay_swin_param], 
                                         'weight_decay': 0.0, 
                                         'lr': lr * coef_lr}, 
                                        {'params': [p for n, p in no_decay_other_param], 
                                         'weight_decay': 0.0}]

        optzr = T.optim.AdamW(optimizer_grouped_parameters, lr=lr, 
                              betas=(0.9, 0.98), weight_decay=weight_decay)
        return optzr

    def step(self, img, txt, mask, ans, is_train):
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step((img, txt, mask, ans))
            out, ans = out
            ls = self.loss_func(out, ans)
        if is_train:
            self.backward_step(ls)
            return ls.item()
        else:
            out = T.argmax(out, dim=1)
            ac = (out==ans).float().tolist()
            return ac

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
        num_ex = len(gathered_ret)
        ret = float(np.average(gathered_ret))

        return ret

    def log_memory(self, ep, step):
        memory = humanbytes(T.cuda.max_memory_allocated())
        lr_swin_bert = f'{self.optzr.param_groups[1]["lr"]:.2e}'
        lr_head = f'{self.optzr.param_groups[0]["lr"]:.2e}'
        return f"ep: {ep}, step: {step}, lr_swin_bert: {lr_swin_bert}, "+f"lr_head: {lr_head}, max memory: {memory}"
    