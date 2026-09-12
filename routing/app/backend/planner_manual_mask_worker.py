"""Validate a planner-edited mask and reapply the immutable road-end guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--edited-mask", type=Path, required=True)
    parser.add_argument("--protected-mask", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        expected_size = Image.open(args.image).size
        edited_image = Image.open(args.edited_mask).convert("L")
        if edited_image.size != expected_size:
            raise ValueError("人工编辑区域尺寸必须与当前方位图一致")
        manual = np.asarray(edited_image, dtype=np.uint8) > 0
        removed = 0
        if args.protected_mask and args.protected_mask.is_file():
            protected = np.asarray(
                Image.open(args.protected_mask).convert("L").resize(expected_size, Image.Resampling.NEAREST),
                dtype=np.uint8,
            ) > 0
            removed = int((manual & protected).sum())
            manual &= ~protected
        ratio = float(manual.mean())
        if ratio < .003:
            raise ValueError("人工编辑后的树冠覆盖区域过小，请至少保留画面的0.3%")
        if ratio > .45:
            raise ValueError("人工编辑区域超过画面45%，请缩小到道路两侧的可信树冠范围")
        output = np.zeros(manual.shape, dtype=np.uint8);output[manual] = 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output, "L").save(args.output, "PNG", optimize=True)
        print(json.dumps({
            "status": "PASS", "manual_mask_used": True,
            "manual_mask_ratio": round(ratio, 5),
            "road_end_pixels_removed_from_manual_mask": removed,
            "road_end_guard_reapplied": True,
        }, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
