"""포스터 CLIP 임베딩 추출 -> {root}/data/features/clip_<encoder>.npy, poster_movie_ids.npy"""
import os

import numpy as np

from _common import parse
from visprofile.encoder import encode_posters, list_posters
from visprofile.evaluate import pick_gpu

args, paths, cfg = parse(__doc__)
os.makedirs(paths.features, exist_ok=True)
files, ids = list_posters(paths.posters)
np.save(paths.movie_ids, ids)
arch, pretrained = cfg.encoder.split("_")
E = encode_posters(paths.posters, files, arch, pretrained, pick_gpu(cfg.min_free_gpu_mib))
np.save(paths.embedding(cfg.encoder), E)
print(cfg.encoder, E.shape)
