
from utils.lib import *
from main_qaoe import VIOLET_QAOE, Dataset_QAOE, Agent_QAOE
from dataset import get_dl
from utils.args import get_args
from utils.logger import LOGGER, add_log_to_file
from utils.dist import NoOp, is_main_process, all_gather, get_rank, get_world_size, iter_tqdm

class Dataset_QAOE_TSV(Dataset_QAOE):
    def __init__(self, args, img_tsv_path, txt, id2lineidx, split, tokzr=None):
        super(Dataset_QAOE, self).__init__(args, split, size_frame=args.size_frame, tokzr=tokzr)
        
        self.txt = txt[split]
        self.img_tsv_path = img_tsv_path
        self.id2lineidx = id2lineidx
        if args.data_ratio!=1: self.get_partial_data()
        if "ans2label" in txt:
            ans2label = txt["ans2label"]
            self.label2ans = {v: k for k, v in ans2label.items()}
        else: self.label2ans = None

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

        txt, mask = self.str2txt(item['question'])
        if video_id not in self.id2lineidx: ans = -1
        else: ans = item['answer']

        return img, txt, mask, ans

class Agent_QAOE_TSV(Agent_QAOE):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        if args.freeze_violet: self.model.freeze()

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
            if is_train: self.log_dict_to_wandb({"train_ls": curr_ret})
            if isinstance(curr_ret, list): ret.extend(curr_ret)
            else: ret.append(curr_ret)

        if (idx%self.args.logging_steps)!=0 and is_train: LOGGER.info(self.log_memory(ep, idx+1))

        gathered_ret = []
        for ret_per_rank in all_gather(ret): gathered_ret.extend(ret_per_rank)
        num_ex = len(gathered_ret)
        ret = float(np.average(gathered_ret))

        return ret
    