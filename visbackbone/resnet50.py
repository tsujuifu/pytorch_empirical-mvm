import torchvision as TV
import torch as T
# from .video_transform import Normalize


class EncImgR50Concat(T.nn.Module):
    def __init__(self, args, hidden_size):
        super().__init__()
        self.imagenet_norm = args.imagenet_norm
        assert args.temporal_fusion == "concat"
        if args.vis_backbone_init == "2d":
            pretrained = True
        else:
            pretrained = False
        self.hidden_size = hidden_size

        res = TV.models.resnet50(pretrained=pretrained)
        for s, p in res.named_parameters():
            if 'layer1' in s or s.startswith('conv1') or s.startswith('bn1'):
                p.requires_grad = not pretrained
        self.emb_img = T.nn.Sequential(*(
            list(res.children())[:-2] + [
                T.nn.Conv2d(2048, hidden_size, 1),
                T.nn.ReLU(True)]))

        self.emb_cls = T.nn.Parameter(0.02*T.randn(1, 1, 1, hidden_size))
        self.emb_pos = T.nn.Parameter(0.02*T.randn(1, 1, 1+14**2, hidden_size))
        self.emb_len = T.nn.Parameter(0.02*T.randn(1, 6, 1, hidden_size))
        self.norm = T.nn.LayerNorm(hidden_size)
        # if args.img_transform == ["vid_rand_crop"]:
        #     self.transform_normalize = None
        # elif self.imagenet_norm:
        #     self.transform_normalize = TV.transforms.Normalize(
        #         [0.485, 0.456, 0.406],
        #         [0.229, 0.224, 0.225])
        # else:
        self.transform_normalize = None

    def forward(self, frame, odr=None, vt_mask=None):
        _B, _T, _C, _H, _W = frame.shape
        _h, _w = _H//32, _W//32

        if self.transform_normalize is not None:
            frame = self.transform_normalize(frame)

        f_frame = self.emb_img(
            frame.reshape([_B*_T, _C, _H, _W])).reshape(
                [_B, _T, self.hidden_size, _h, _w])
        f_frame = f_frame.permute(0, 1, 3, 4, 2).reshape(
            [_B, _T, _h*_w, self.hidden_size])
        f_frame = T.cat(
            [self.emb_cls.expand([_B, _T, -1, -1]), f_frame],
            dim=2)
        f_frame += self.emb_pos.expand([_B, _T, -1, -1])[:, :, :1+_h*_w, :]

        if odr is not None:
            emb_len = []  # feed order
            for b in range(_B):
                tmp = T.cat([
                    self.emb_len[:, i:i+1, :, :]
                    if i == p else self.emb_odr
                    for i, p in enumerate(odr[b])], dim=1)
                emb_len.append(tmp)
            emb_len = T.cat(emb_len, dim=0)
            f_frame += emb_len
        else:
            f_frame += self.emb_len.expand([_B, -1, 1+_h*_w, -1])[:, :_T, :, :]

        f_frame = self.norm(f_frame)

        m_frame = T.ones(1+_h*_w).long().unsqueeze(0).unsqueeze(0).expand(
            [_B, _T, -1]).cuda()

        f_frame = f_frame.reshape([_B, _T*(1+_h*_w), -1])
        m_frame = m_frame.reshape([_B, _T*(1+_h*_w)])

        return f_frame, m_frame


class EncImgR50Mean(T.nn.Module):
    def __init__(self, args, hidden_size):
        super().__init__()
        self.imagenet_norm = args.imagenet_norm
        assert args.temporal_fusion == "mean"
        if args.vis_backbone_init == "2d":
            pretrained = True
        else:
            pretrained = False
        self.hidden_size = hidden_size

        res = TV.models.resnet50(pretrained=pretrained)
        for s, p in res.named_parameters():
            if 'layer1' in s or s.startswith('conv1') or s.startswith('bn1'):
                p.requires_grad = not pretrained
        self.emb_img = T.nn.Sequential(*(
            list(res.children())[:-2] + [
                T.nn.Conv2d(2048, hidden_size, 1),
                T.nn.ReLU(True)]))

        self.emb_cls = T.nn.Parameter(
            0.02*T.randn(1, 1, 1, hidden_size))
        self.emb_pos = T.nn.Parameter(
            0.02*T.randn(1, 1, 1+14**2, hidden_size))
        self.emb_len = T.nn.Parameter(
            0.02*T.randn(1, 6, 1, hidden_size))
        self.norm = T.nn.LayerNorm(hidden_size)
        # if args.img_transform == ["vid_rand_crop"]:
        #     self.transform_normalize = None
        # elif self.imagenet_norm:
        #     self.transform_normalize = TV.transforms.Normalize(
        #         [0.485, 0.456, 0.406],
        #         [0.229, 0.224, 0.225])
        # else:
        self.transform_normalize = None

    def forward(self, img, odr=None, vt_mask=None):
        _B, _T, _C, _H, _W = img.shape
        _h, _w = _H//32, _W//32

        if self.transform_normalize is not None:
            img = self.transform_normalize(img)

        f_img = self.emb_img(
            img.reshape([_B*_T, _C, _H, _W])).reshape(
                [_B, _T, self.hidden_size, _h*_w])
        f_img = f_img.view(
            [_B, _T, self.hidden_size, _h*_w]).permute(0, 1, 3, 2)

        f_img = T.mean(f_img, dim=1, keepdim=True)
        _T = 1  # MEAN

        f_img = T.cat([self.emb_cls.expand([_B, _T, -1, -1]), f_img], dim=2)
        f_img += self.emb_pos.expand(
            [_B, _T, -1, -1])[:, :, :1+_h*_w, :]
        f_img += self.emb_len.expand([_B, -1, 1+_h*_w, -1])[:, :_T, :, :]
        f_img = self.norm(f_img).view([_B, _T*(1+_h*_w), -1])

        m_img = T.ones(1+_h*_w).long().cuda().unsqueeze(0).unsqueeze(0)
        m_img = m_img.expand([_B, _T, -1]).contiguous().view(
            [_B, _T*(1+_h*_w)])

        return f_img, m_img
