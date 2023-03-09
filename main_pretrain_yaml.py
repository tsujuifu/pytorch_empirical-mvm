
from utils.lib import *
from dataset import TsvCompositeDataset, make_data_loader, MetaLoader
from utils.args import get_args
from utils.logger import LOGGER, RunningMeter, add_log_to_file
from utils.dist import is_main_process, get_rank, get_world_size, NoOp, iter_tqdm
from main_pretrain import VIOLET_Pretrain, Agent_Pretrain
from collections import Counter

class Dataset_Pretrain_YAML(TsvCompositeDataset):
    def __init__(self, args, yaml_file, split, size_frame, tokzr=None):
        super().__init__(args, yaml_file, split, size_frame=size_frame, tokzr=tokzr)
        
        if "vq" in args.mvm_target:
            if args.dalle_model_path is not None and op.exists(args.dalle_model_path): LOGGER.info(f"MVM-VQ: Extracting VQ tokens on-the-fly for {yaml_file}")
            else: raise ValueError(f"Load pre-extracted vq is disabled")

    def get_img_txt_pair(self, idx):
        img_idx, cap_idx = self.get_image_cap_index(idx)
        img_key = self.image_keys[img_idx]
        caption_sample, tag, start, end, _ = self.get_caption_and_timeinfo_wrapper(img_idx, cap_idx)
        frames, is_video = self.get_visual_data(img_idx)

        if isinstance(caption_sample, dict): caption = caption_sample["caption"]
        else:
            caption = caption_sample
            caption_sample = None
            
        meta_data = {}
        meta_data['caption'] = caption
        meta_data['img_key'] = img_key
        meta_data['is_video'] = is_video
        meta_data['tag'] = tag
        meta_data['img'] = frames
        return meta_data

    def get_visual_data(self, idx):
        row = self.get_row_from_tsv(self.visual_tsv, idx)

        if len(row)>=(self.size_frame+2): return self.get_img_or_video(row[2:]), True
        elif len(row)==(self.size_frame+1): return self.get_img_or_video(row[1:]), True
        else: return self.get_img_or_video([row[-1]]), False

    @property
    def vtm_prompt_text(self):
        return "is the video-text paired, true or false?"

    def get_vtm_prompt(self):
        return self.get_prompt(prompt_text=self.vtm_prompt_text)

    @property
    def cap_prompt_text(self):
        return "write a description about the video."

    def get_cap_prompt(self):
        return self.get_prompt(prompt_text=self.cap_prompt_text)

    def __getitem__(self, idx):
        try: raw_data = self.get_img_txt_pair(idx)
        except Exception as e: print(e, self.yaml_file)
        img = raw_data['img']
        raw_txt = raw_data['caption']
        txt, mask = self.str2txt(raw_txt)
        vid = raw_data['img_key']
        if "hog" in self.args.mvm_target: hog = self.get_hog_features(img)
        else: hog = None
        return img, txt, mask, vid, hog

    def collate_batch(self, inputs):
        img, txt, mask, vid, hog = map(list, unzip(inputs))
        all_imgs = T.stack(img, dim=0)

        all_txts = T.stack(txt, dim=0)
        all_masks = T.stack(mask, dim=0)
        all_vqs = None
        if hog[0] is not None: all_hogs = T.stack(hog, dim=0)
        else: all_hogs = None

        batch = {"img": all_imgs, "txt": all_txts, "mask": all_masks, "vq": all_vqs, "hog": all_hogs, 'vid': vid}
        return batch

class Agent_Pretrain_YAML(Agent_Pretrain):
    def __init__(self, args, model):
        super().__init__(args, model)
        
        self.task2loss = {}
        self.log = defaultdict(list)
        self.ds_tr_steps = defaultdict(int)

    def meter_loss(self, dataset, ls):
        for key, val in ls.items():
            ls_key = f'{dataset}_ls_{key}'
            if ls_key not in self.task2loss: self.task2loss[ls_key] = RunningMeter(ls_key)
            self.task2loss[ls_key](val)

    def log_train(self):
        log_info = self.log_memory()
        log_info += "\n\t"
        for task, rm in self.task2loss.items():
            ls_tr = rm.val
            ls_tr = f'{ls_tr:.6f}' if ls_tr!=-1 else -1
            log_info += f" {task}: {ls_tr}"
            self.log_dict_to_wandb({f'train_{task}': rm.val})
        return log_info

    def go_ep(self, dl_trs, dl_vls, ep):
        for ds_tr_key, dl_tr in dl_trs.items():
            iter_per_ep = self.args.iter_per_ep[ds_tr_key]
            eval_step = self.args.eval_step[ds_tr_key]
            step = 0
            global_step = (ep-1)*iter_per_ep+step
            dl_tr.start_iter = global_step
            for _, batch in enumerate(dl_tr):
                if step==0: print(f"first_batch: {batch['vid'][0]}")
                batch = defaultdict(lambda: None, batch)
                if (step%self.args.logging_steps)==0: LOGGER.info(f'Train dataset {ds_tr_key}: '
                                                                  f'{self.log_train()}')
                img, txt, mask, vq = [batch[key] for key in ["img", "txt", "mask", "vq"]]
                masked_batch = self.masking(img, txt, mask, vq)
                batch.update(masked_batch)
                batch = self.prepare_batch(batch)
                ls = self.step(batch, is_train=True)
                self.meter_loss(ds_tr_key, ls)
                step += 1
                self.global_step += 1
                if (step%eval_step==0) and step:
                    for ds_vl_key, dl_vl in dl_vls.items():
                        res_vl = self.evaluate(dl_vl)
                        for k in res_vl:
                            self.log[f'{ds_vl_key}_vl_{k}'].append(res_vl[k])
                            self.log_dict_to_wandb({f'{ds_vl_key}_vl_{k}': res_vl[k]})
                        LOGGER.info(f'Train dataset {ds_tr_key}, '
                                    f'ep {ep}, step {step}, '
                                    f'{ds_vl_key} vl: {json.dumps(res_vl)}')
                    self.save_model(ep, ds_tr_key, step)
                if step>=iter_per_ep: break

            if (step%self.args.logging_steps)!=0: LOGGER.info(f'Train dataset {ds_tr_key}:'+self.log_train())

            if (step%eval_step)!=0:
                for ds_vl_key, dl_vl in dl_vls.items():
                    res_vl = self.evaluate(dl_vl)
                    for k in res_vl:
                        self.log[f'{ds_vl_key}_acc_{k}'].append(res_vl[k])
                        self.log_dict_to_wandb({f'{ds_vl_key}_vl_{k}': res_vl[k]})
                    LOGGER.info(f'Train dataset {ds_tr_key},Ep {ep}, step {step}, '
                                f'{ds_vl_key} vl: {json.dumps(res_vl)}')
                self.save_model(ep, ds_tr_key, step)
        return

    def run_meta_loader(self, dl_trs, dl_vls):
        LOGGER.info("Start training....")
        step = 0

        for step, (ds_tr_key, batch) in enumerate(dl_trs):
            ep = step//self.args.iter_per_ep
            self.ds_tr_steps[ds_tr_key] += 1
            if (step%self.args.logging_steps)==0: LOGGER.info(self.log_train()+f'\n\t\t {self.ds_tr_steps}')

            batch = defaultdict(lambda: None, batch)
            img, txt, mask, vq = [batch[key] for key in ["img", "txt", "mask", "vq"]]
            masked_batch = self.masking(img, txt, mask, vq)
            batch.update(masked_batch)
            batch = self.prepare_batch(batch)
            ls = self.step(batch, is_train=True)
            self.global_step += 1
            self.meter_loss(ds_tr_key, ls)
            if (step%self.args.eval_step)==0 and step:
                for ds_vl_key, dl_vl in dl_vls.items():
                    res_vl = self.evaluate(dl_vl)
                    for k in res_vl:
                        self.log[f'{ds_vl_key}_vl_{k}'].append(res_vl[k])
                        self.log_dict_to_wandb({f'{ds_vl_key}_vl_{k}': res_vl[k]})
                    LOGGER.info(f'Ep {ep+1}, step {step}, '
                                f'{ds_vl_key} vl: {json.dumps(res_vl)}')
                self.save_model(ep+1, '', step)
            if step>=self.args.max_iter: break
        if (step%self.args.logging_steps)==0: LOGGER.info(self.log_train()+f'\n\t\t {self.ds_tr_steps}')

        if (step%self.args.eval_step)!=0 and step:
            for ds_vl_key, dl_vl in dl_vls.items():
                res_vl = self.evaluate(dl_vl)
                for k in res_vl:
                    self.log[f'{ds_vl_key}_acc_{k}'].append(res_vl[k])
                    self.log_dict_to_wandb({f'{ds_vl_key}_vl_{k}': res_vl[k]})
                LOGGER.info(f'Ep {ep}, step {step}, '
                            f'{ds_vl_key} vl: {json.dumps(res_vl)}')
            self.save_model(ep+1, '', step)

    def run(self, dl_trs, dl_vl):
        if not isinstance(dl_trs, MetaLoader):
            LOGGER.info("Start training....")
            for ep in iter_tqdm(range(self.args.size_epoch)): self.go_ep(dl_trs, dl_vl, ep+1)
        else: self.run_meta_loader(dl_trs, dl_vl)

    def evaluate(self, dl):
        self.model.eval()
        ret = defaultdict(list)
        for _, batch in enumerate(dl):
            batch = defaultdict(lambda: None, batch)
            img, txt, mask, vq = [batch[key] for key in ["img", "txt", "mask", "vq"]]
            masked_batch = self.masking(img, txt, mask, vq)
            batch.update(masked_batch)
            if self.args.enable_prompt:
                batch["vtm_prompt"] = dl.dataset.get_vtm_prompt()
                batch["cap_prompt"] = dl.dataset.get_cap_prompt()
            batch = self.prepare_batch(batch)
            r = self.step(batch, is_train=False)
            ret = {k: ret[k]+[l] for k, l in r.items()}

        ret = {k: self.reduce_mean(float(np.average([v for v in l if not math.isnan(v)]))) \
               for k, l in ret.items()}
        self.model.train()
        return ret

if __name__=='__main__':
    args = get_args()
    args.task += f"-{args.dataset}"
    args.path_output = '%s/_%s_%s'%(args.path_output, args.task, datetime.now().strftime('%Y%m%d%H%M%S'))
    print(args)
    
    LOGGER.info("Loading Data....")
    tokzr = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    ds_vl = {}
    for key, val_yaml in args.val_yaml.items():
        if key in ['coco', 'sbu', 'vg', 'cc3m', 'cc12m']: size_frame = 1
        else: size_frame = args.size_frame
        ds = Dataset_Pretrain_YAML(args, val_yaml, 'val', size_frame, tokzr=tokzr)
        ds_vl[key] = ds
    dl_vls = {key: make_data_loader(args, ds)[0] for key, ds in ds_vl.items()}

    if args.size_epoch:
        dl_trs = {}
        dl_trs_len = {}
        args.images_per_batch = {}
        args.iter_per_ep = {}
        args.max_iter = 0
        args.eval_step = {}
        for key, tr_yaml in args.train_yaml.items():
            if key in ['coco', 'sbu', 'vg', 'cc3m', 'cc12m']: size_frame = 1
            else: size_frame = args.size_frame
            ds = Dataset_Pretrain_YAML(args, tr_yaml, 'train', size_frame, tokzr=tokzr)
            dl_trs_len[key] = len(ds)
            dl_tr, info_ = make_data_loader(args, ds)
            images_per_batch, iter_per_ep, num_iters = info_
            args.images_per_batch[key] = images_per_batch
            args.iter_per_ep[key] = iter_per_ep
            args.max_iter += num_iters
            size_part = args.size_part[key] if key in args.size_part else 1
            args.eval_step[key] = min(iter_per_ep, max(20, iter_per_ep//size_part))
            dl_trs[key] = dl_tr

        LOGGER.info(f"#Examples for each dataset {dl_trs_len}")
        LOGGER.info(f"Training steps per epoch for each dataset {args.iter_per_ep}")

        min_iter_per_ep = max(20, min(list(args.iter_per_ep.values())))
        meta_dl_trs = {}
        for key, dl in dl_trs.items(): meta_dl_trs[key] = (dl, args.iter_per_ep[key]//min_iter_per_ep)
        dl_trs = MetaLoader(meta_dl_trs, distributed=args.distributed)
        if isinstance(dl_trs, MetaLoader):
            args.iter_per_ep = sum(list(args.iter_per_ep.values()))
            args.eval_step = min(args.iter_per_ep, sum(list(args.eval_step.values())))
            LOGGER.info(f"MetaLoader Sampling Pool {Counter(dl_trs.sampling_pools)}")
        LOGGER.info(f"Total batch size {args.images_per_batch}")
        LOGGER.info(f"Total training steps {args.max_iter}")
        LOGGER.info(f"Training steps per epoch (accumulated) {args.iter_per_ep}")
        LOGGER.info(f"Eval steps (accumulated) {args.eval_step}")
    else:
        dl_trs = None
        args.max_iter = 1

    model = VIOLET_Pretrain(args, tokzr)
    model.load_ckpt(args.path_ckpt)
    model.cuda()
    if args.distributed: LOGGER.info(f"n_gpu: {args.num_gpus}, rank: {get_rank()},"
                                     f" world_size: {get_world_size()}")

    agent = Agent_Pretrain_YAML(args, model)
    if args.distributed: agent.prepare_dist_model()
    agent.save_training_meta()
    if is_main_process(): add_log_to_file('%s/stdout.txt' % (args.path_output))
    else: LOGGER = NoOp()
    LOGGER.info("Saved training meta infomation ...")

    agent.setup_wandb()
    if os.path.exists(args.path_ckpt):
        LOGGER.info("Zero shot evaluation ...")
        for ds_vl_key, dl_vl in dl_vls.items():
            res_vl = agent.evaluate(dl_vl)
            for k in res_vl: agent.log[f'{ds_vl_key}_vl_{k}'].append(res_vl[k])
            LOGGER.info(f'ZS eval, '
                        f'{ds_vl_key} vl: {json.dumps(res_vl)}')
    else: LOGGER.info("No pretrained ckpt, skip zero shot evaluation ...")

    if args.size_epoch: agent.run(dl_trs, dl_vls)
        