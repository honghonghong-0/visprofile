import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visprofile.config import Config, Paths  # noqa: E402


def parse(desc, extra=None):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--root", default=os.environ.get("VISPROFILE_ROOT", "."), help="데이터/결과 루트")
    ap.add_argument("--encoder", default=Config.encoder, help="예: ViT-L-14_openai, ViT-B-16_openai")
    if extra:
        extra(ap)
    args = ap.parse_args()
    return args, Paths(args.root), Config(encoder=args.encoder)
