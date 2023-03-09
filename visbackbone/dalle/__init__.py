import io, requests
import torch
import torch.nn as nn
import os
from torchvision.transforms import Normalize

from .encoder import Encoder
from .decoder import Decoder
from .utils import map_pixels, unmap_pixels


def load_model(path: str, device: torch.device = None) -> nn.Module:
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(path)
        resp.raise_for_status()
        with io.BytesIO(resp.content) as buf:
            return torch.load(buf, map_location=device)
    else:
        with open(path, 'rb') as f:
            return torch.load(f, map_location=device)


class DalleModel(nn.Module):
    def __init__(self, pretrained_path, size_img, denorm=True):
        super().__init__()
        assert os.path.exists(pretrained_path)
        self.encoder = load_model(pretrained_path, "cpu")
        self.size_img = size_img
        if denorm:
            # hardcoded
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
            self.unnormalize = Normalize(
                (-mean / std).tolist(), (1.0 / std).tolist())
        else:
            self.unnormalize = None

    def preprocess(self, img):
        if self.unnormalize is not None:
            img = self.unnormalize(img)
        img = map_pixels(img)
        return img

    def extract_vq_token(self, img):
        self.encoder.eval()
        orig_dtype = img.dtype
        encoder = self.encoder
        if orig_dtype != torch.float:
            img = img.to(dtype=torch.float)
            encoder = self.encoder.float()
        img = self.preprocess(img)
        z_logits = encoder(img)
        vq_tokens = torch.argmax(z_logits, axis=1)
        return vq_tokens

    def get_vq_patch_size(self):
        # hardcoded
        return 8
