
from utils.lib import *
from dataset import Dataset_Base, get_dl
from model import VIOLET_Base
from agent import Agent_Base
from utils.dist import iter_tqdm
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import is_main_process, get_rank, get_world_size, iter_tqdm, NoOp
from visbackbone.optical_flow import raft_large
from visbackbone.dalle import DalleModel
from visbackbone.midas.dpt_depth import DPTDepthModel
import copy

class Dataset_Pretrain(Dataset_Base):
    def __init__(self, args, txt, vq, dataset, split, part=None, data_dir=None, tokzr=None):
        super().__init__(args, split=split, size_frame=args.size_frame, tokzr=tokzr)
        
        if dataset in ["cc3m", "coco", "vg", "cc12m", "sbu"]: self.size_frame = 1
        if dataset in ["coco", "vg", "sbu"]:
            if part is not None and part>1: raise ValueError(f"Double check size_part for {dataset}")
        self.dataset, self.part = dataset, part
        if data_dir is not None: self.data_dir = data_dir
        else: self.data_dir = args.data_dir

        self.txt = txt[self.split]
        if vq is not None and args.dalle_model_path is not None and op.exists(args.dalle_model_path):
            LOGGER.info("MVM-VQ: Extracting VQ tokens on-the-fly, "+f"skip loading pre-extractred vqs.....")
            self.vq = None
        else: self.vq = vq
        
        if self.dataset=="shutterstock12m": self.lineidx = [int(p) \
                                                            for p in open(f'{self.data_dir}/_shutterstock-tsv_frame4/shutterstock12m-{self.part+1:03d}.img.lineidx' \
                                                                          if self.split=='train' \
                                                                          else f'{self.data_dir}/webvid2.5m_val.lineidx', 'r')]
        elif self.dataset=="shutterstock12m_filtered": self.lineidx = [int(p) \
                                                                       for p in open(f'{self.data_dir}/image-1{self.part:04d}.lineidx' \
                                                                                     if self.split=='train' \
                                                                                     else f'{self.data_dir}/webvid2.5m_val.lineidx', 'r')]
        elif self.dataset=="cc12m": self.lineidx = [int(p) \
                                                    for p in open(f'{self.data_dir}/train.{self.part}.62.img.lineidx' \
                                                                  if self.split=='train' \
                                                                  else f'{self.data_dir}/cc3m_val.lineidx', 'r')]
        elif self.dataset in ["cc3m", "webvid2.5m"]: self.lineidx = [int(p) \
                                                                     for p in open(f'{self.data_dir}/{self.dataset}_train_{self.part}.lineidx' \
                                                                                   if self.split=='train' \
                                                                                   else f'{self.data_dir}/{self.dataset}_val.lineidx', 'r')]
        else: self.lineidx = [int(p) \
                              for p in open(f'{self.data_dir}/_{self.dataset}/train.img.lineidx' \
                                            if self.split=='train' \
                                            else f'{self.data_dir}/_{self.dataset}/val.img.lineidx', 'r')]

    def read_tsv(self, worker_id):
        if self.dataset=="shutterstock12m": self.tsv = open(f'{self.data_dir}/_shutterstock-tsv_frame4/shutterstock12m-{self.part+1:03d}.img.tsv' \
                                                            if self.split=='train' \
                                                            else f'{self.data_dir}/webvid2.5m_val.tsv', 'r')
        elif self.dataset=="shutterstock12m_filtered": self.tsv = open(f'{self.data_dir}/image-1{self.part:04d}.tsv' \
                                                                       if self.split=='train' \
                                                                       else f'{self.data_dir}/webvid2.5m_val.tsv', 'r')
        elif self.dataset=="cc12m": self.tsv = open(f'{self.data_dir}/train.{self.part}.62.img.tsv' \
                                                    if self.split=='train' \
                                                    else f'{self.data_dir}/cc3m_val.tsv', 'r')
        else: self.tsv = open(f'{self.data_dir}/{self.dataset}_train_{self.part}.tsv' \
                              if self.split=='train' \
                              else f'{self.data_dir}/{self.dataset}_val.tsv', 'r')

    def __len__(self):
        return len(self.lineidx)

    def __getitem__(self, idx):
        corrupt = False
        lineidx = self.lineidx[idx]
        self.tsv.seek(lineidx)
        item = self.tsv.readline().split('\t')

        if self.dataset in ["shutterstock12m", "shutterstock12m_filtered"] and self.split=="train": vid, bufs = item[0], item[2:]
        else: vid, bufs = item[0], item[1:]

        if vid in self.txt: raw_txt = self.txt[vid][0]
        else:
            print(f"Failed to load txt for video {vid} for ",
                  f"dataset {self.dataset}, split {self.split}, ",
                  f"part {self.part}")
            corrupt = True

        if self.vq is None: vq = None
        elif vid in self.vq: vq = T.from_numpy(np.array(sum([[-1]+c.flatten().tolist() for c in self.vq[vid]], []), dtype=np.int64))
        else:
            vq = None
            print(f"Failed to load vq for video {vid} for ",
                  f"dataset {self.dataset}, split {self.split}, ",
                  f"part {self.part}")
            corrupt = True
        try:
            img = self.get_img_or_video(bufs)
            (_T, _, _H, _W) = img.shape
        except Exception as e:
            print(f"Failed to load image binaries for video {vid} for ",
                  f"dataset {self.dataset}, split {self.split}, ",
                  f"part {self.part}")
            _T = self.args.size_frame
            _H = self.args.size_img
            _W = _H
            _C = 3
            img = T.zeros((_T, _C, _H, _W))
            corrupt = True
        _h, _w = _H//self.args.size_patch, _W//self.args.size_patch

        if "hog" in self.args.mvm_target: hog = self.get_hog_features(img)
        else: hog = None

        if vq is None: vq = T.ones(_T*(_h*_w+1)).long()*-1

        if corrupt:
            raw_txt = ""
            img = T.zeros_like(img)
            vq = T.ones(_T*(_h*_w+1)).long()*-1

        txt, mask = self.str2txt(raw_txt)

        return img, txt, mask, vq, hog

    def collate_batch(self, inputs):
        try:
            img, txt, mask, vq, hog = map(list, unzip(inputs))
        except Exception as e:
            img, txt, mask, vq = map(list, unzip(inputs))
            hog = [None]
        all_imgs = T.stack(img, dim=0)
        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)
        if vq[0] is not None: all_vqs = T.stack(vq, dim=0)
        else: all_vqs = None
        if hog[0] is not None: all_hogs = T.stack(hog, dim=0)
        else: all_hogs = None

        batch = {"img": all_imgs, "txt": all_txts, "mask": all_masks, "vq": all_vqs, "hog": all_hogs}
        return batch

class VIOLET_Pretrain(VIOLET_Base):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)
        
        self.size_vq = 8192
        self.patch_size = args.size_patch
        self.fc = T.nn.Sequential(*[T.nn.Dropout(0.1),  T.nn.Linear(self.hidden_size, self.hidden_size*2), T.nn.ReLU(inplace=True), 
                                    T.nn.Linear(self.hidden_size*2, 1)])
        bert = transformers.AutoModelForMaskedLM.from_pretrained(self.args.tokenizer)
        if isinstance(bert, transformers.RobertaForMaskedLM): self.fc_mtm = bert.lm_head
        else: self.fc_mtm = bert.cls
        del bert

        if "3d_feature" in args.mvm_target:
            from visbackbone.video_swin import get_vidswin_model
            feature_args = copy.deepcopy(args)
            feature_args["vis_backbone_init"] = "3d"
            feature_args["vis_backbone_size"] = "base"
            feature_args["kinetics"] = 600
            self.feature_model = get_vidswin_model(feature_args)
            feat_size = self.feature_model.norm.normalized_shape[0]
            LOGGER.info("MVM-Feature: MVM with 3d feature value from video swin as gt")
            self.fc_mvm = T.nn.Sequential(*[T.nn.Dropout(0.1), T.nn.Linear(self.hidden_size, self.hidden_size*2), T.nn.ReLU(inplace=True), 
                                            T.nn.Linear(self.hidden_size*2, feat_size)])
        if "2d_feature" in args.mvm_target:
            from visbackbone.swin import get_swin_model
            feature_args = copy.deepcopy(args)
            feature_args["vis_backbone_init"] = "2d"
            feature_args["vis_backbone_size"] = "base"
            feature_args["imagenet"] = 22
            self.feature_model = get_swin_model(feature_args)
            feat_size = self.feature_model.num_features
            LOGGER.info("MVM-Feature: MVM with 2d feature value from  swin as gt")
            self.fc_mvm = T.nn.Sequential(*[T.nn.Dropout(0.1), T.nn.Linear(self.hidden_size, self.hidden_size*2), T.nn.ReLU(inplace=True), 
                                            T.nn.Linear(self.hidden_size*2, feat_size)])
        if "pixel" in args.mvm_target:
            from visbackbone.video_swin import get_vidswin_model
            LOGGER.info("MVM-Pixel: MVM with pixel value as gt")
            self.decoder_pixel = T.nn.Sequential(T.nn.Conv2d(in_channels=self.hidden_size, out_channels=(self.patch_size**2)*3, kernel_size=1), 
                                                 T.nn.PixelShuffle(self.patch_size))
        if "hog" in args.mvm_target:
            LOGGER.info("MVM-HOG: Computing HOG on-the-fly")
            self.decoder_hog = T.nn.Sequential(T.nn.Conv2d(in_channels=self.hidden_size, out_channels=self.patch_size**2, kernel_size=1), 
                                               T.nn.PixelShuffle(self.patch_size))
        if "optical_flow" in args.mvm_target:
            self.raft = raft_large(pretrained=True, progress=False)
            LOGGER.info("MVM-OF: Computing optical flow on-the-fly")
            self.decoder_flow = T.nn.Sequential(T.nn.Conv2d(in_channels=self.hidden_size*2, out_channels=(self.patch_size**2)*2, kernel_size=1), 
                                                T.nn.PixelShuffle(self.patch_size))
        if "depth" in args.mvm_target:
            self.dpt = DPTDepthModel(path="models/midas/dpt_large-midas-2f21e586.pt", backbone="vitl16_384", non_negative=True)
            LOGGER.info("MVM-Depth: Computing depth on-the-fly")
            self.decoder_depth = T.nn.Sequential(T.nn.Conv2d(in_channels=self.hidden_size, out_channels=self.patch_size**2, kernel_size=1), 
                                                 T.nn.PixelShuffle(self.patch_size))
        if "vq" in args.mvm_target:
            if args.dalle_model_path is not None and op.exists(args.dalle_model_path):
                LOGGER.info("MVM-VQ: Extracting VQ tokens on-the-fly, "+f"pretrained dalle loaded from {args.dalle_model_path}")
                self.dalle = DalleModel(args.dalle_model_path, size_img=args.size_img)
                vq_patch_size = self.dalle.get_vq_patch_size()
                upscale_factor = self.patch_size//vq_patch_size
                self.decoder_vq = T.nn.Sequential(T.nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size*2, kernel_size=1), 
                                                  T.nn.PixelShuffle(upscale_factor))
                self.vq_pred_channel_size = (self.hidden_size*2//upscale_factor//upscale_factor)
            else:
                LOGGER.info(f"MVM-VQ: Use pre-extracted VQ tokens, from 56x56 images")
                self.dalle = None
                self.decoder_vq = None
                self.vq_pred_channel_size = self.hidden_size
            self.fc_mvm = T.nn.Sequential(*[T.nn.Dropout(0.1), T.nn.Linear(self.vq_pred_channel_size, self.vq_pred_channel_size*2), T.nn.ReLU(inplace=True), 
                                            T.nn.Linear(self.vq_pred_channel_size*2, self.size_vq)])

    def get_att(self, img, txt, mask, odr=None):
        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask, odr)
        _, att = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        att = T.cat([a.mean(dim=1, keepdim=True) for a in att], dim=1).sum(dim=(1, 2))
        return feat_img, att

    def get_smtm_output(self, feat_img, mask_img, feat_txt, mask_txt, ans_mtm):
        feat = T.cat([feat_img, feat_txt], dim=1)
        mask = self.get_attn_mask(mask_img, mask_txt, attn_mask_type="seq2seq")
        mask = self.mask_ext(mask, mask.shape, mask.device)
        mask = mask.to(dtype=feat_img.dtype)
        out = self.trsfr(feat, mask, output_attentions=True)
        out_smtm = out['last_hidden_state']
        return out_smtm, ans_mtm

    def forward(self, batch):
        img, txt, mask, ans_mtm, ans_mvm = [batch[key] for key in ["img", "txt", "mask", "ans_mtm", "ans_mvm"]]
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//self.patch_size, _W//self.patch_size
        _O = min(_B, 4)

        feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
        out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
        if self.args.temporal_fusion=="mean": _T = 1

        out_mtm = self.fc_mtm(out[:, (1+_h*_w)*_T:])
        out_mvm = out[:, :(1+_h*_w)*_T]
        if "smtm" in self.args.pretrain_tasks:
            out_smtm, ans_smtm = self.get_smtm_output(feat_img, mask_img, feat_txt, mask_txt, ans_mtm)
            out_smtm = self.fc_mtm(out_smtm[:, (1+_h*_w)*_T:])
        else: out_smtm, ans_smtm = None, None

        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [], [], [], []
        for i in range(_B):
            pdt_feat_img.append(feat_img[i].unsqueeze(0))
            pdt_mask_img.append(mask_img[i].unsqueeze(0))
            pdt_feat_txt.append(feat_txt[i].unsqueeze(0))
            pdt_mask_txt.append(mask_txt[i].unsqueeze(0))

            neg = np.random.permutation([j for j in range(_B) if j!=i])
            for j in range(_O-1):
                j = neg[j]
                pdt_feat_img.append(feat_img[i].unsqueeze(0))
                pdt_mask_img.append(mask_img[i].unsqueeze(0))
                pdt_feat_txt.append(feat_txt[j].unsqueeze(0))
                pdt_mask_txt.append(mask_txt[j].unsqueeze(0))
        pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt = [T.cat(x, dim=0) \
                                                                  for x in [pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt]]
        out, _ = self.go_cross(pdt_feat_img, pdt_mask_img, pdt_feat_txt, pdt_mask_txt)
        out_vtm = self.fc(out[:, (1+_h*_w)*_T, :]).squeeze().view([_B, _O]) / self.args.temp

        ans_vtm = T.tensor([0 for _ in range(_B)]).long().cuda()

        output = {"out_vtm": out_vtm, "out_mvm": out_mvm, "out_mtm": out_mtm,
                  "out_smtm": out_smtm, "ans_vtm": ans_vtm, "ans_mtm": ans_mtm,
                  "ans_mvm": ans_mvm, "ans_smtm": ans_smtm}
        return output

class Agent_Pretrain(Agent_Base):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        self.patch_size = self.model.patch_size
        self.log = {dataset: defaultdict(list) for dataset in self.args.dataset}

    @T.no_grad()
    def masking(self, img, txt, mask, vq, p_mask=0.15):
        orig_img = img.clone()
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H//self.patch_size, _W//self.patch_size
        if vq is None:
            v_mask = T.from_numpy(np.array(sum([[-1]+[0]*_h*_w for _ in range(_T)], []), dtype=np.int64))
            v_mask = v_mask.repeat(_B, 1)
        else: v_mask = vq
        spc_txt, spc_v = [T.logical_or(T.logical_or(txt==self.cls_token_id, txt==self.sep_token_id), 
                                      T.logical_or(txt==self.pad_token_id, txt==self.mask_token_id)), 
                          v_mask==-1]
        spc_all = T.cat([spc_v, spc_txt], dim=1)

        ans_mtm = T.ones(txt.shape).long()*-1
        ans_mvm = []
        mvm_mask = []

        if p_mask<=0:
            ans_mvm = T.ones(v_mask.shape).long()*-1
            mvm_mask = T.zeros(img.shape).long()
            return {"img": img, "txt": txt, "mask": mask, 
                    "ans_mtm": ans_mtm, "ans_mvm": ans_mvm, 
                    "mvm_mask": mvm_mask, "unmask_img": orig_img}

        failed_masking = False
        for i in range(_B):
            mask_type = random.choice(self.args.pretrain_masks)
            if mask_type=="bm":
                mask_mtm = T.where(T.logical_and(T.logical_not(spc_txt[i]), T.rand(_X)<p_mask))[0]
                mask_mvm = set()
                if "mvm" in self.args.pretrain_tasks:
                    for _ in range(_T):
                        t, h, w = [np.random.randint(1, _T) if _T>1 else 1, 
                                   np.random.randint(1, _h*2//3), 
                                   np.random.randint(1, _w*2//3)]
                        t1, h1, w1 = [np.random.randint(0, _T-t+1), 
                                      np.random.randint(0, _h-h+1), 
                                      np.random.randint(0, _w-w+1)]
                        for i_t in range(t1, t1+t):
                            for i_h in range(h1, h1+h):
                                for i_w in range(w1, w1+w): mask_mvm.add((i_t, i_h, i_w))
                mask_mvm = list(mask_mvm)

            if mask_type=="am":
                with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
                    model_instance = self.model.module if hasattr(self.model, "module") else self.model
                    _, att = model_instance.get_att(*self.prepare_batch((img, txt, mask)))
                if self.args.temporal_fusion=="mean":
                    spc_all = spc_txt
                    _T_feat = 1
                else: _T_feat = _T
                att[i][T.where(spc_all[i])] = 0.0

                try:
                    pos = T.multinomial(att[i], int(((1+_h*_w)*_T_feat+_X)*p_mask)).data.cpu().numpy()
                    mask_mtm, mask_mvm = [], []
                    for p in pos:
                        if p<(1+_h*_w)*_T_feat:
                            if "mvm" in self.args.pretrain_tasks:
                                i_t, p = p//(1+_h*_w), p%(1+_h*_w)-1
                                i_h, i_w = p//_w, p%_w
                                mask_mvm.append((i_t, i_h, i_w))
                        else:
                            p -= (1+_h*_w)*_T_feat
                            mask_mtm.append(p)
                    if "mvm" in self.args.pretrain_tasks: failed_masking = len(mask_mtm)==0
                except Exception: failed_masking = True
            if mask_type=="rm" or failed_masking:
                mask_mtm = T.where(T.logical_and(T.logical_not(spc_txt[i]), T.rand(_X)<p_mask))[0]
                mask_mvm = []
                if "mvm" in self.args.pretrain_tasks:
                    v_pos = T.where(T.logical_and(T.logical_not(spc_v[i]), T.rand((1+_h*_w)*_T)<p_mask))[0]
                    for p in v_pos:
                        i_t, p = p//(1+_h*_w), p%(1+_h*_w)-1
                        i_h, i_w = p//_w, p%_w
                        mask_mvm.append((i_t, i_h, i_w))

            for p in mask_mtm: ans_mtm[i][p], txt[i][p] = txt[i][p], self.mask_token_id

            cov = T.zeros(_T, _h, _w)
            curr_ans_mvm = T.ones(v_mask[i].shape).long()*-1
            for i_t, i_h, i_w in mask_mvm:
                cov[i_t][i_h][i_w] = 1.0
                p = (1+_h*_w)*i_t+1+i_h*_w+i_w
                if vq is not None: curr_ans_mvm[p] = vq[i][p]
            cov = cov.unsqueeze(1).unsqueeze(3).unsqueeze(5).expand([-1, 3, -1, 32, -1, 32])
            cov = cov.flatten(2, 3).flatten(3, 4)
            img[i] *= (1.0-cov)
            
            ans_mvm.append(curr_ans_mvm)
            mvm_mask.append(cov)
        ans_mvm = T.stack(ans_mvm, dim=0)
        mvm_mask = T.stack(mvm_mask, dim=0)
        return {"img": img, "txt": txt, "mask": mask,
                "ans_mtm": ans_mtm, "ans_mvm": ans_mvm,
                "mvm_mask": mvm_mask, "unmask_img": orig_img}

    def calc_mvm_loss(self, batch, out_mvm, is_train=True):
        mvm_loss_type = getattr(self.args, "mvm_loss", "l1")
        ls_mvm = {}
        if "mvm" not in self.args.pretrain_tasks:
            if is_train: return None
            else: return ls_mvm

        img = batch["unmask_img"]
        _B, _T, _in_C, _H, _W = img.shape
        mvm_mask = batch["mvm_mask"]
        model_instance = self.model.module if hasattr(self.model, "module") else self.model

        if _T>1 and "optical_flow" in self.args.mvm_target:
            _B, _T, _in_C, _H, _W = img.shape
            _h, _w = _H//self.patch_size, _W//self.patch_size
            _, _L, _C = out_mvm.shape
            _l = _L // _T
            non_cls_out_mvm = T.cat([out_mvm[:, _l*_t+1: _l*(_t+1), :] for _t in range(_T)], dim=1)
            non_cls_out_mvm = non_cls_out_mvm.permute(0, 2, 1).reshape(_B, _C, _T, _h, _w)
            
            non_cls_out_mvm_1 = non_cls_out_mvm[:, :, :-1, :, :].contiguous()
            non_cls_out_mvm_2 = non_cls_out_mvm[:, :, 1:, :, :].contiguous()
            non_cls_out_mvm = T.cat([non_cls_out_mvm_1, non_cls_out_mvm_2], dim=1)

            non_cls_out_mvm = non_cls_out_mvm.permute(0, 2, 1, 3, 4).reshape(_B*(_T-1), _C*2, _h, _w)
            non_cls_out_mvm = model_instance.decoder_flow(non_cls_out_mvm).view(_B, (_T-1), 2, _H, _W)

            img_1 = img[:, :-1, :, :, :].contiguous().view(-1, _in_C, _H, _W)
            img_2 = img[:, 1:, :, :, :].contiguous().view(-1, _in_C, _H, _W)
            with T.no_grad():
                with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
                    model_instance.raft.eval()
                    list_of_flows = model_instance.raft(img_1, img_2)
                    target = list_of_flows[-1].view(_B, _T-1, 2, _H, _W)
                    target_mask = target.view(_B, _T-1, -1)
                    target_mask = T.max(T.abs(target_mask), dim=-1)[0]
                    target = target.to(dtype=img.dtype)
                    mvm_mask_flow = mvm_mask[:, :-1, :, :, :]+mvm_mask[:, 1:, :, :, :]
                    mvm_mask_flow = (mvm_mask_flow.sum(dim=2)>0).view(_B, (_T-1), 1, _H, _W).expand_as(non_cls_out_v)
                    target_mask = (target_mask<50.).view(_B, (_T-1), 1, 1, 1)
                    mvm_mask_flow = mvm_mask_flow*target_mask

            ls_mvm_flow = T.nn.functional.l1_loss(non_cls_out_mvm, target, reduction='none')
            ls_mvm_flow = (ls_mvm_flow.float()*mvm_mask_flow.float()).sum()/(mvm_mask_flow.float().sum()+1e-5)/2
            if not is_train: ls_mvm_flow = float(ls_mvm_flow.item())
            ls_mvm["mvm_flow"] = ls_mvm_flow
        if "pixel" in self.args.mvm_target:
            _B, _T, _in_C, _H, _W = img.shape
            _h, _w = _H//self.patch_size, _W//self.patch_size
            _, _L, _C = out_mvm.shape
            _l = _L//_T
            non_cls_out_mvm = T.cat([out_mvm[:, _l*_t+1: _l*(_t+1), :] for _t in range(_T)], dim=1)
            non_cls_out_mvm = non_cls_out_mvm.permute(0, 2, 1).reshape(_B, _C, _T, _h, _w)
            non_cls_out_mvm = non_cls_out_mvm.permute(0, 2, 1, 3, 4).view(_B*_T, _C, _h, _w)
            non_cls_out_mvm = model_instance.decoder_pixel(non_cls_out_mvm).view(_B, _T, _in_C, _H, _W)
            ls_mvm_pixel = T.nn.functional.l1_loss(non_cls_out_mvm, img, reduction='none')
            ls_mvm_pixel = (ls_mvm_pixel.float()*mvm_mask.float()).sum()/(mvm_mask.float().sum()+1e-5)/_in_C
            if not is_train: ls_mvm_pixel = float(ls_mvm_pixel.item())
            ls_mvm["mvm_pixel"] = ls_mvm_pixel
        if "depth" in self.args.mvm_target:
            _B, _T, _in_C, _H, _W = img.shape
            _h, _w = _H//self.patch_size, _W//self.patch_size
            _, _L, _C = out_mvm.shape
            _l = _L//_T
            non_cls_out_mvm = T.cat([out_mvm[:, _l*_t+1: _l*(_t+1), :] for _t in range(_T)], dim=1)
            non_cls_out_mvm = non_cls_out_mvm.permute(0, 2, 1).reshape(_B, _C, _T, _h, _w)
            non_cls_out_mvm = non_cls_out_mvm.permute(0, 2, 1, 3, 4).view(_B*_T, _C, _h, _w)
            non_cls_out_mvm = model_instance.decoder_depth(non_cls_out_mvm).view(_B, _T, 1, _H, _W)
            with T.no_grad():
                with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
                    model_instance.dpt.eval()
                    img_input = img.view(_B* _T, _in_C, _H, _W)
                    target = model_instance.dpt(img_input)
                    target = target.view(_B, _T, 1, target.size(-2), target.size(-1))
                    target = target.to(dtype=img.dtype)
            ls_mvm_depth = T.nn.functional.l1_loss(non_cls_out_mvm, target, reduction='none')
            ls_mvm_depth = (ls_mvm_depth.float()*mvm_mask.float()).sum()/(mvm_mask.float().sum()+1e-5)/_in_C
            if not is_train: ls_mvm_depth = float(ls_mvm_depth.item())
            ls_mvm["mvm_depth"] = ls_mvm_depth
        if "hog" in self.args.mvm_target:
            hog = batch["hog"]
            assert hog is not None
            _B, _T, _H, _W = hog.shape
            _h, _w = _H//self.patch_size, _W//self.patch_size
            _, _L, _C = out_mvm.shape
            _l = _L//_T
            non_cls_out_mvm = T.cat([out_mvm[:, _l*_t+1: _l*(_t+1), :] for _t in range(_T)], dim=1)
            non_cls_out_mvm = non_cls_out_mvm.permute(0, 2, 1).reshape(_B, _C, _T, _h, _w)
            non_cls_out_mvm = non_cls_out_mvm.permute(0, 2, 1, 3, 4).view(_B*_T, _C, _h, _w)
            non_cls_out_mvm = model_instance.decoder_hog(non_cls_out_mvm).view(_B, _T, _H, _W)
            ls_mvm_hog = T.nn.functional.l1_loss(non_cls_out_mvm, hog, reduction='none')
            mvm_mask_hog = (mvm_mask.sum(dim=2)>0)
            ls_mvm_hog = (ls_mvm_hog.float()*mvm_mask_hog.float()).sum()/(mvm_mask_hog.float().sum()+1e-5)
            if not is_train: ls_mvm_hog = float(ls_mvm_hog.item())
            ls_mvm["mvm_hog"] = ls_mvm_hog
        if 'vq' in self.args.mvm_target:
            dalle = model_instance.dalle
            vq_patch_size = dalle.get_vq_patch_size()
            if dalle is not None:
                _B, _T, _in_C, _H, _W = img.shape
                _h, _w = _H//self.patch_size, _W//self.patch_size
                _, _L, _C = out_mvm.shape
                _l = _L//_T
                non_cls_out_mvm = T.cat([out_mvm[:, _l*_t+1: _l*(_t+1), :] for _t in range(_T)], dim=1)
                non_cls_out_mvm = non_cls_out_mvm.permute(0, 2, 1).reshape(_B, _C, _T, _h, _w)

                with T.no_grad():
                    with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
                        img_input_to_dalle = img.view(_B*_T, _in_C, _H, _W)
                        vq_tokens = dalle.extract_vq_token(img_input_to_dalle)
                        vq_size = vq_tokens.shape[-1]
                        assert vq_size==(_H//vq_patch_size)
                        vq_tokens = vq_tokens.long()
                        mvm_mask = mvm_mask.view(_B*_T, _in_C, _H, _W)
                        mvm_mask_vq = T.nn.functional.max_pool2d(mvm_mask, vq_patch_size).sum(dim=1)
                        ans_mvm = T.where(mvm_mask_vq==0, -1, vq_tokens)
                        ans_mvm = ans_mvm.view(_B, _T*vq_size*vq_size).detach()

                non_cls_out_mvm = non_cls_out_mvm.permute(0, 2, 1, 3, 4).reshape(_B*_T, _C, _h, _w)
                non_cls_out_mvm = model_instance.decoder_vq(non_cls_out_mvm).view(_B, _T, -1, vq_size, vq_size)
                vq_out_mvm = non_cls_out_mvm.permute(0, 1, 3, 4, 2)
                vq_out_mvm = vq_out_mvm.view(_B, _T, vq_size*vq_size, -1)
                vq_out_mvm = vq_out_mvm.contiguous().view(_B, _T*vq_size*vq_size, -1)
            else:
                vq_out_mvm = out_mvm
                ans_mvm = batch["ans_mvm"]
            out_mvm = model_instance.fc_mvm(vq_out_mvm)
            if is_train:
                ls_mvm_vq = self.loss_func(out_mvm.flatten(0, len(out_mvm.shape)-2), ans_mvm.flatten(0, len(ans_mvm.shape)-1))
                ls_mvm["mvm_vq"] = ls_mvm_vq
            else:
                out_mvm = T.argmax(out_mvm, dim=-1)
                ac_mvm = float((out_mvm==ans_mvm).sum()/(ans_mvm!=-1).sum()) if (ans_mvm!=-1).sum()>0 else 0
                ls_mvm["mvm_vq"] = ac_mvm
        if '3d_feature' in self.args.mvm_target:
            _B, _T, _in_C, _H, _W = img.shape
            _h, _w = _H//self.patch_size, _W//self.patch_size
            _, _L, _C = out_mvm.shape
            _l = _L//_T
            non_cls_out_mvm = T.cat([out_mvm[:, _l*_t+1: _l*(_t+1), :] for _t in range(_T)], dim=1)
            non_cls_out_mvm = model_instance.fc_mvm(non_cls_out_mvm).reshape(_B, _T, _h*_w, -1)
            model_instance.feature_model.eval()
            with T.no_grad():
                with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
                    f_img = model_instance.feature_model(img.transpose(1, 2)).transpose(1, 2)
                    target = f_img.permute(0, 1, 3, 4, 2).view([_B, _T, _h*_w, -1])
                    mvm_mask = mvm_mask.view(_B*_T, _in_C, _H, _W)
                    mvm_mask = T.nn.functional.max_pool2d(mvm_mask, self.patch_size).sum(dim=1)/3.
                    mvm_mask = mvm_mask.view(_B, _T, _h*_w, 1)
            ls_mvm_feature = T.nn.functional.l1_loss(non_cls_out_mvm, target, reduction='none')
            ls_mvm_feature = (ls_mvm_feature.float()*mvm_mask.float()).sum()/(mvm_mask.float().sum()+1e-5)/_in_C
            if not is_train: ls_mvm_feature = float(ls_mvm_feature.item())
            ls_mvm["mvm_3d_feature"] = ls_mvm_feature
        if '2d_feature' in self.args.mvm_target:
            _B, _T, _in_C, _H, _W = img.shape
            _h, _w = _H//self.patch_size, _W//self.patch_size
            _, _L, _C = out_mvm.shape
            _l = _L//_T
            non_cls_out_mvm = T.cat([out_mvm[:, _l*_t+1: _l*(_t+1), :] for _t in range(_T)], dim=1)
            non_cls_out_mvm = model_instance.fc_mvm(non_cls_out_mvm).reshape(_B, _T, _h*_w, -1)
            model_instance.feature_model.eval()
            with T.no_grad():
                with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
                    f_img = model_instance.feature_model(img.flatten(0, 1), output_hidden_states=True)['hidden_states'][-1]
                    target = f_img.permute(0, 2, 1).view([_B, _T, -1, _h*_w]).permute(0, 1, 3, 2)
                    mvm_mask = mvm_mask.view(_B*_T, _in_C, _H, _W)
                    mvm_mask = T.nn.functional.max_pool2d(mvm_mask, self.patch_size).sum(dim=1)/3.
                    mvm_mask = mvm_mask.view(_B, _T, _h*_w, 1)
            ls_mvm_feature = T.nn.functional.l1_loss(non_cls_out_mvm, target, reduction='none')
            ls_mvm_feature = (ls_mvm_feature.float()*mvm_mask.float()).sum()/(mvm_mask.float().sum()+1e-5)/_in_C
            if not is_train: ls_mvm_feature = float(ls_mvm_feature.item())
            ls_mvm["mvm_3d_feature"] = ls_mvm_feature
        if len(ls_mvm):
            if is_train:
                loss = 0
                for v in ls_mvm.values(): loss += v
                ls_mvm = loss
            return ls_mvm
        else:
            return None

    def step(self, batch, is_train=True):
        with T.cuda.amp.autocast(enabled=not self.args.deepspeed):
            out = self.forward_step(batch)
            out_mtm, out_mvm, out_vtm, out_smtm = (out[key] for key in ["out_mtm", "out_mvm", "out_vtm", "out_smtm"])
            ans_mtm, ans_mvm, ans_vtm, ans_smtm = (out[key] for key in ["ans_mtm", "ans_mvm", "ans_vtm", "ans_smtm"])
            ls_mtm = self.loss_func(out_mtm.flatten(0, len(out_mtm.shape)-2), ans_mtm.flatten(0, len(ans_mtm.shape)-1))
            ls_vtm = self.loss_func(out_vtm.flatten(0, len(out_vtm.shape)-2), ans_vtm.flatten(0, len(ans_vtm.shape)-1))

            ls_mvm = self.calc_mvm_loss(batch, out_mvm, is_train=True)
            ls = ls_mtm+ls_vtm
            if ls_mvm is not None: ls += ls_mvm
            if out_smtm is not None:
                ls_smtm = self.loss_func(out_smtm.flatten(0, len(out_smtm.shape)-2), ans_mtm.flatten(0, len(ans_mtm.shape)-1))
                ls += ls_smtm

        if is_train:
            self.backward_step(ls)
            return {'mtm': ls_mtm.item(), 'mvm': ls_mvm.item() if ls_mvm is not None else -1, 
                    'vtm': ls_vtm.item(), "smtm": ls_smtm.item() if out_smtm is not None else -1}
        else:
            out_mtm, out_vtm = [T.argmax(o, dim=-1) for o in [out_mtm, out_vtm]]

            ac_mtm, ac_vtm = [float((o==a).sum()/(a!=-1).sum()) if (a!=-1).sum()>0 else -1 \
                              for o, a in zip([out_mtm, out_vtm], [ans_mtm, ans_vtm])]
            ac_mvm = self.calc_mvm_loss(batch, out_mvm, is_train=False)
            res = {'mtm': ac_mtm, 'vtm': ac_vtm}
            if ac_mvm is not None: res.update(ac_mvm)
            if out_smtm is not None:
                out_smtm = T.argmax(out_smtm, dim=-1)
                ac_smtm = float((out_smtm==ans_mtm).sum()/(ans_mtm!=-1).sum()) if (ans_mtm!=-1).sum()>0 else -1
                res.update({"smtm": ac_smtm})
            return res

    def go_dl(self, ep, dl, is_train):
        if is_train: self.model.train()
        else: self.model.eval()
        ret = defaultdict(list)

        idx = 0
        for idx, batch in enumerate(dl):
            if is_train: self.global_step += 1
            batch = defaultdict(lambda: None, batch)
            if (idx%self.args.logging_steps)==0 and is_train: LOGGER.info(self.log_memory())
            img, txt, mask, vq = [batch[key] for key in ["img", "txt", "mask", "vq"]]
            masked_batch = self.masking(img, txt, mask, vq)
            batch.update(masked_batch)
            batch = self.prepare_batch(batch)
            r = self.step(batch, is_train)
            ret = {k: ret[k]+[l] for k, l in r.items()}
            if is_train: self.log_dict_to_wandb({f'train_{k}': l for k, l in r.items()})

        if (idx%self.args.logging_steps)!=0 and is_train: LOGGER.info(self.log_memory())

        ret = {k: self.reduce_mean(float(np.average([v for v in l if not math.isnan(v)]))) \
               for k, l in ret.items()}
        return ret

    def save_model(self, ep, dataset="init", part=0):
        if is_main_process():
            output_dir = self.args.path_output
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            state_dict = {k: v.cpu() if isinstance(v, T.Tensor) else v for k, v in model_to_save.state_dict().items()}
            T.save(state_dict, os.path.join(f"{self.args.path_output}/"
                                            f"ckpt_violet_pretrain_{dataset}_{part}_{ep}.pt"))
            