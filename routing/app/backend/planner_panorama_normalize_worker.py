"""Normalize an uploaded panorama to the Agent's 2048x512 geometry on CUDA."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required; CPU image-processing fallback forbidden")
    rgb=torch.from_numpy(np.asarray(Image.open(args.input).convert("RGB"),dtype=np.uint8).copy()).permute(2,0,1).unsqueeze(0).to("cuda",dtype=torch.float32)
    with torch.inference_mode():normalized=F.interpolate(rgb,(512,2048),mode="bilinear",align_corners=False,antialias=True)[0].round().clamp(0,255).to(torch.uint8)
    args.output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(normalized.permute(1,2,0).cpu().numpy(),"RGB").save(args.output,compress_level=3);return 0


if __name__=="__main__":raise SystemExit(main())
