
from utils.lib import *
from dataset import Dataset_Base, get_dl
from model import VIOLET_Base
from agent import Agent_Base, NormSoftmaxLoss
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import is_main_process, get_rank, get_world_size, iter_tqdm, NoOp

class Dataset_Retrieval(Dataset_Base):
    def __init__(self, args, img, txt, split, tokzr=None):
        super().__init__(args, split, size_frame=args.size_frame, tokzr=tokzr)
        
        self.img, self.txt = img, txt[split]
        if args.data_ratio!=1: self.get_partial_data()

    def img_transform_rand_crop(self, img):
        if self.split=="train": img = TV.transforms.Compose([TV.transforms.Resize(self.args.size_img), 
                                                             TV.transforms.RandomCrop((self.args.size_img, self.args.size_img)), 
                                                             TV.transforms.RandomHorizontalFlip(p=0.5), 
                                                             TV.transforms.ToTensor()])(img)
        else: img = TV.transforms.Compose([TV.transforms.Resize(self.args.size_img), 
                                           TV.transforms.CenterCrop((self.args.size_img, self.args.size_img)), 
                                           TV.transforms.ToTensor()])(img)
        return img

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, idx):
        item = self.txt[idx]
        vid = item['video']
        img = self.get_img_or_video(self.img[vid])

        raw_txt = item['caption']
        if isinstance(raw_txt, list):
            sent_ids = range(len(raw_txt))
            if self.split=="train":
                size_sent = random.randint(1, len(raw_txt))
                sent_ids = random.sample(sent_ids, size_sent)
            raw_txt = " ".join([raw_txt[i] for i in sent_ids])

        txt, mask = self.str2txt(raw_txt)

        return img, txt, mask, vid

    def collate_batch(self, inputs):
        img, txt, mask, video_id = map(list, unzip(inputs))

        all_imgs = T.stack(img, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)

        batch = {"img": all_imgs, "txt": all_txts, "mask": all_masks, "vid": video_id}
        return batch

class VIOLET_Retrieval(VIOLET_Base):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)
        
        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.1), T.nn.Linear(self.hidden_size, self.hidden_size*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(self.hidden_size*2, 1)])

    def forward(self, img, txt, mask, vid):
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//32, _W//32

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)

        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [], [], [], []
        for i in range(_B):
            for j in range(_B):
                pdt_feat_img.append(feat_img[i].unsqueeze(0))
                pdt_mask_img.append(mask_img[i].unsqueeze(0))
                pdt_feat_txt.append(feat_txt[j].unsqueeze(0))
                pdt_mask_txt.append(mask_txt[j].unsqueeze(0))
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [T.cat(x, dim=0) \
                                                                  for x in [pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt]]
        out, _ = self.go_cross(pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt)
        if self.args.temporal_fusion=="mean": _T = 1
        out = self.fc(out[:, (1+_h*_w)*_T, :]).squeeze().view([_B, _B])

        ans = T.tensor([i for i in range(_B)]).long().cuda()

        return out, ans

class Agent_Retrieval(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        self.loss_func = NormSoftmaxLoss(temperature=args.temp).cuda()
        self.log = {'ls_tr': [], 'ac_vl': [], 'ac_ts': []}
        if args.freeze_violet: self.model.freeze()

    def step(self, img, txt, mask, vid, is_train):
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step((img, txt, mask, vid))
            out, ans = out
            ls = self.loss_func(out)
        if is_train:
            self.backward_step(ls)
            return ls.item()
        else:
            out = T.argmax(out, dim=1)
            ac = (out==ans).float().mean().item()
            return ac

    def go_dl(self, ep, dl, is_train):
        if is_train: self.model.train()
        else: self.model.eval()
        ret = []
        idx = 0
        for idx, batch in iter_tqdm(enumerate(dl)):
            if is_train: self.global_step += 1
            if (idx%self.args.logging_steps)==0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))
            img, txt, mask, vid = self.prepare_batch(batch)
            curr_ret = self.step(img, txt, mask, vid, is_train)
            ret.append(curr_ret)

        if (idx%self.args.logging_steps)!=0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))

        ret = float(float(np.average(ret)))
        if self.args.distributed: ret = self.reduce_mean(ret)
        return ret
    