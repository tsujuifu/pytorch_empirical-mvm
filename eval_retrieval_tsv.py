
from utils.lib import *
from utils.args import get_args
from eval_retrieval import VIOLET_RetrievalEval, Dataset_Product, Dataset_Retrieval

class Dataset_Retrieval_TSV(Dataset_Retrieval):
    def __init__(self, args, split):
        super(Dataset_Retrieval, self).__init__(args, split, size_frame=args.size_frame)

        self.img_tsv_path = f'{args.data_dir}/img_{args.dataset}.tsv'
        self.id2lineidx = pickle.load(open(f'{args.data_dir}/img_{args.dataset}.id2lineidx.pkl', 'rb'))
        self.txt = json.load(open(f'{args.data_dir}/txt_{args.task}.json', 'r'))[split]
        self.gt_txt2vid = {idx: item["video"] for idx, item in enumerate(self.txt)}

    def __getitem__(self, idx):
        item = self.txt[idx]

        video_id = item['video']
        lineidx = self.id2lineidx[video_id]
        b = self.seek_img_tsv(lineidx)[2:]
        img = self.get_clips_with_temporal_sampling(b)

        raw_txt = item['caption']
        if isinstance(raw_txt, list):
            assert self.split!="train"
            raw_txt = " ".join(raw_txt)

        txt, mask = self.str2txt(raw_txt)

        return img, txt, mask, idx, item['video']

if __name__=='__main__':
    args = get_args(distributed=False)
    args.use_checkpoint = False
    args.num_gpus = T.cuda.device_count()
    args.multi_clip_testing = True
    if args.multi_clip_testing: args.size_batch = 10*args.num_gpus
    else: args.size_batch = 100*args.num_gpus
    assert os.path.exists(args.path_ckpt)
    print(args)
    
    ds_ret = Dataset_Retrieval_TSV(args, "test")

    log = {}
    model = T.nn.DataParallel(VIOLET_RetrievalEval(args, ds_ret.tokzr).cuda())
    model.module.load_ckpt(args.path_ckpt)
    model.eval()

    for split in ['test']:
        ds_ret = Dataset_Retrieval_TSV(args, split)
        dl = T.utils.data.DataLoader(ds_ret, batch_size=args.size_batch, 
                                     shuffle=False, num_workers=32, pin_memory=True, worker_init_fn=ds_ret.read_tsv)
        featv, featt = {}, {}
        gt_txt2vid = ds_ret.gt_txt2vid
        for img, txt, mask, tid, vid in tqdm(dl, ascii=True):
            with T.no_grad(): feat_img, mask_img, feat_txt, mask_txt = model(typ='feat', img=img.cuda(), txt=txt.cuda(), mask=mask.cuda())
            for t, v, f_i, m_i, f_t, m_t in zip(tid, vid, *[d.data.cpu().numpy() for d in [feat_img, mask_img, feat_txt, mask_txt]]):
                if v not in featv: featv[v] = {'video': v, 'feat_img': f_i, 'mask_img': m_i}
                featt[t] = {'tid': t, 'feat_txt': f_t, 'mask_txt': m_t}
        ds = Dataset_Product(featv, featt)
        dl = T.utils.data.DataLoader(ds, batch_size=args.size_batch,
                                     shuffle=False, num_workers=args.n_workers, pin_memory=True)
        print(f"number of videos: {len(ds.vid2idx)}")
        print(f"number of queires (by text): {len(ds.tid2idx)}")
        print(f"number of queries (before gathering rank): {len(ds_ret)}")
        rank = {}
        for feat_txt, mask_txt, tid, feat_img, mask_img, vid in tqdm(dl, ascii=True):
            with T.no_grad():
                out = model(typ='cross', feat_img=feat_img, mask_img=mask_img, feat_txt=feat_txt, mask_txt=mask_txt)
                out = T.sigmoid(out).data.cpu().numpy()
            for tid_, vid_, o in zip(tid, vid, out):
                i_v = ds.vid2idx[vid_]
                i_v, o = int(i_v), float(o)
                tid_ = tid_.item()

                if tid_ not in rank: rank[tid_] = []
                rank[tid_].append([i_v, o])

        res = {'r@1': 0, 'r@5': 0, 'r@10': 0, 'median': []}
        print(f"number of queries (after gathering rank): {len(rank)}")
        for tid_ in rank:
            tmp = sorted(rank[tid_], key=lambda d: -d[1])
            gt_iv = ds.vid2idx[gt_txt2vid[tid_]]
            p = [d[0] for d in tmp].index(gt_iv)+1

            if p<=1: res['r@1'] += 1.0/len(rank)
            if p<=5: res['r@5'] += 1.0/len(rank)
            if p<=10: res['r@10'] += 1.0/len(rank)
            res['median'].append(p)
        res['median'] = int(np.median(res['median']))
        res = {key: f"{val*100:.2f}" if key!='median' else f"{val}" for key, val in res.items()}
        log[split] = res
        print(split, res)
        