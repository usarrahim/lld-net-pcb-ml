"""Run LLD-Net on a PCB image or folder.

Examples
--------
python infer.py board.jpg
python infer.py images/ --model baseline --conf 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

from project_paths import setup


def main() -> None:
    paths = setup()
    weights = {
        "p2": paths.p2_best,
        "baseline": paths.baseline_best,
    }

    parser = argparse.ArgumentParser(description="LLD-Net PCB defect inference")
    parser.add_argument("source", help="Image file or directory")
    parser.add_argument(
        "--model",
        choices=sorted(weights),
        default="p2",
        help="Checkpoint to load (default: p2)",
    )
    parser.add_argument("--weights", default=None, help="Override checkpoint path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="cuda:0 or cpu")
    parser.add_argument(
        "--out",
        default="runs_pcb/infer",
        help="Directory for annotated predictions",
    )
    args = parser.parse_args()

    ckpt = Path(args.weights) if args.weights else weights[args.model]
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=str(Path(args.out).parent),
        name=Path(args.out).name,
        exist_ok=True,
    )
    n = len(results)
    print(f"{n} image(s) → {args.out}  [{args.model}: {ckpt.name}]")


if __name__ == "__main__":
    main()
