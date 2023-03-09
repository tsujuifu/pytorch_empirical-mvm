
from utils.lib import *
from dataset import Dataset_Base, get_dl
from model import VIOLET_Base
from agent import Agent_Base
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import NoOp, is_main_process, all_gather, get_rank, get_world_size, iter_tqdm

class Dataset_QAOE(Dataset_Base):
    def __init__(self, args, img, txt, split, tokzr=None):
        super().__init__(args, split, size_frame=args.size_frame, tokzr=tokzr)
        
        self.img, self.txt = img, txt[split]
        if args.data_ratio!=1: self.get_partial_data()
        ans2label = txt["ans2label"]
        self.label2ans = {v: k for k, v in ans2label.items()}

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, idx):
        item = self.txt[idx]

        img = self.get_img_or_video(self.img[item['video']])

        txt, mask = self.str2txt(item['question'])

        return img, txt, mask, item['answer']

    def collate_batch(self, inputs):
        img, txt, mask, ans = map(list, unzip(inputs))

        all_imgs = T.stack(img, dim=0)
        all_ans = T.LongTensor(ans)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)

        batch = {"img": all_imgs, "txt": all_txts, "mask": all_masks, "ans": all_ans}
        return batch

class VIOLET_QAOE(VIOLET_Base):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)
        
        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.1), T.nn.Linear(self.hidden_size, self.hidden_size*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(self.hidden_size*2, args.size_vocab)])

    def forward(self, img, txt, mask, ans):
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        if self.args.temporal_fusion == "mean": _T = 1
        out = self.fc(out[:, (1+_h*_w)*_T, :])

        return out, ans

    def reinit_head(self):
        del self.fc
        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.1), T.nn.Linear(self.hidden_size, self.hidden_size*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(self.hidden_size*2, args.size_vocab)])

class Agent_QAOE(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        self.log = defaultdict(list)
        if args.freeze_violet: self.model.freeze()

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
        for idx, batch in enumerate(dl):
            if is_train: self.global_step += 1
            if (idx%self.args.logging_steps)==0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))
            img, txt, mask, ans = self.prepare_batch(batch)
            curr_ret = self.step(img, txt, mask, ans, is_train)
            if isinstance(curr_ret, list): ret.extend(curr_ret)
            else: ret.append(curr_ret)

        if (idx%self.args.logging_steps)!=0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))

        gathered_ret = []
        for ret_per_rank in all_gather(ret): gathered_ret.extend(ret_per_rank)
        num_ex = len(gathered_ret)
        ret = float(np.average(gathered_ret))

        return ret
    