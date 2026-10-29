"""포스터 이미지 -> CLIP 임베딩.

세로형 포스터는 CLIP 기본 전처리(중앙 정사각형 자르기)에서 상하(제목부)가 잘리므로,
정사각형 여백을 채운 뒤 전처리한다.
"""
import os

import numpy as np
import torch
from PIL import Image

PAD_COLOR = (123, 117, 104)  # CLIP 픽셀 평균 근사: 정규화 후 여백이 중립값이 되도록


def pad_square(im):
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), PAD_COLOR)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


class PosterDataset(torch.utils.data.Dataset):
    def __init__(self, poster_dir, files, preprocess):
        self.dir, self.files, self.preprocess = poster_dir, files, preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.dir, self.files[i])).convert("RGB")
        return self.preprocess(pad_square(im))


def list_posters(poster_dir):
    files = sorted((f for f in os.listdir(poster_dir) if f.endswith(".jpg")), key=lambda f: int(f[:-4]))
    return files, np.array([int(f[:-4]) for f in files])


@torch.no_grad()
def encode_posters(poster_dir, files, arch, pretrained, device, batch_size=512, workers=8):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
    model.eval()
    loader = torch.utils.data.DataLoader(PosterDataset(poster_dir, files, preprocess),
                                         batch_size=batch_size, num_workers=workers)
    out = []
    with torch.autocast("cuda", dtype=torch.float16):
        for x in loader:
            e = model.encode_image(x.to(device)).float()
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
    return torch.cat(out).numpy().astype(np.float32)
