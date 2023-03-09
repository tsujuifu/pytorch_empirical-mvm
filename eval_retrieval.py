
from utils.lib import *
from dataset import Dataset_Base
from utils.args import get_args
from main_retrieval import VIOLET_Retrieval

class Dataset_Retrieval(Dataset_Base):
    def __init__(self, args, split):
        super().__init__(args, split, size_frame=args.size_frame)

        self.img = pickle.load(open(f'{args.data_dir}/img_{args.dataset}.pkl', 'rb'))
        self.txt = json.load(open(f'{args.data_dir}/txt_{args.task}.json', 'r'))[split]
        self.gt_txt2vid = {idx: item["video"] for idx, item in enumerate(self.txt)}

    def __len__(self):
        return len(self.txt)

    def get_clips_with_temporal_sampling(self, list_of_b):
        max_size_frame = len(list_of_b)

        list_of_sampled_videos = []
        if max_size_frame==1 or self.size_frame==max_size_frame:
            list_of_sampled_videos.append(self.get_img_or_video(list_of_b).unsqueeze(0))
            return T.cat(list_of_sampled_videos, dim=0)

        if max_size_frame<self.size_frame: print(f"Error in size_frame", f"\tasked for {size_frame} from {max_size_frame} frames")

        size_frame = min(self.size_frame, max_size_frame)
        size_clips = int(math.ceil(max_size_frame/size_frame))
        if self.args.multi_clip_testing:
            for sampled_start in range(size_clips):
                sampled_end = min(sampled_start+(size_frame-1)*size_clips, max_size_frame-1)
                sampled_index = self.sampling(sampled_start, sampled_end, size_frame)
                sampled_video = [list_of_b[i] for i in sampled_index]
                sampled_video = self.get_img_or_video(sampled_video)
                list_of_sampled_videos.append(sampled_video.unsqueeze(0))
        else:
            sampled_index = self.sampling(0, max_size_frame-1, size_frame)
            sampled_video = [list_of_b[i] for i in sampled_index]
            sampled_video = self.get_img_or_video(sampled_video)
            list_of_sampled_videos.append(sampled_video.unsqueeze(0))
        list_of_sampled_videos = T.cat(list_of_sampled_videos, dim=0)
        return list_of_sampled_videos

    def get_img_or_video(self, bufs):
        img = []
        for b in bufs:
            single_img = self.str2img(b)
            if self.args.img_transform==["vid_rand_crop"]:
                vis_transform = "vid_center_crop"
                img.append(single_img)
            else:
                if self.args.img_transform==["pad_resize"]:
                    vis_transform = "pad_resize"
                    single_img = self.pad_resize(single_img)
                else:
                    vis_transform = "img_center_crop"
                    single_img = self.img_center_crop(single_img)
                img.append(single_img.unsqueeze(0))
        if vis_transform=="vid_center_crop": img = self.vid_center_crop(img)
        else: img = T.cat(img, dim=0)
        return img

    def __getitem__(self, idx):
        item = self.txt[idx]

        img = self.get_clips_with_temporal_sampling(self.img[item['video']])

        raw_txt = item['caption']
        if isinstance(raw_txt, list):
            assert self.split!="train"
            raw_txt = " ".join(raw_txt)

        txt, mask = self.str2txt(raw_txt)

        return img, txt, mask, idx, item['video']

class Dataset_Product(T.utils.data.Dataset):
    def __init__(self, featv, featt):
        super().__init__()
        self.vids = list(set([item["video"] for key, item in featv.items()]))
        self.vid2idx = {v: i for i, v in enumerate(self.vids)}
        self.tids = list(set([item["tid"] for key, item in featt.items()]))
        self.tid2idx = {t: i for i, t in enumerate(self.tids)}
        self.lst = [[featt[p], featv[q]] for p in featt for q in featv]

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        p, q = self.lst[idx]

        return [p['feat_txt'], p['mask_txt'], p['tid'],
                q['feat_img'], q['mask_img'], q['video']]

class VIOLET_RetrievalEval(VIOLET_Retrieval):
    def __init__(self, args, tokzr=None):
        super().__init__(args, tokzr)

    def forward(self, typ, img=None, txt=None, mask=None, feat_img=None, mask_img=None, feat_txt=None, mask_txt=None):
        if typ == 'feat':
            _B, _Clips, _T, _C, _H, _W = img.shape
            img = img.view(-1, _T, _C, _H, _W)
            feat_img, mask_img, feat_txt, mask_txt = self.go_feat(img, txt, mask)
            _hidden_size = feat_img.shape[-1]
            mean_mask_img = mask_img.view(_B, _Clips, -1)
            mean_feat_img = feat_img.view(_B, _Clips, -1, _hidden_size)
            mean_feat_img = mean_feat_img.mean(dim=1)
            mean_mask_img = mean_mask_img[:, 0, :]
            return mean_feat_img, mean_mask_img, feat_txt, mask_txt

        elif typ=='cross':
            out, _ = self.go_cross(feat_img, mask_img, feat_txt, mask_txt)
            out = self.fc(out[:, feat_img.shape[1], :]).squeeze()
            return out
        