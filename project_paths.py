"""Resolve repo, dataset, and run directories for Colab and local use.

Notebooks import ``setup()`` so the same cells work on Google Colab
(Drive + local SSD cache) and on a laptop checkout.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

COLAB_REPO = Path("/content/drive/MyDrive/lld-net-pcb-ml")
MARKER = Path("cfg") / "yolo12n_p2.yaml"


def find_repo_root() -> Path:
    if IN_COLAB:
        return COLAB_REPO
    here = Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / MARKER).exists():
            return candidate
    fallback = Path(__file__).resolve().parent
    if (fallback / MARKER).exists():
        return fallback
    raise FileNotFoundError(
        "Could not find the repo root (missing cfg/yolo12n_p2.yaml). "
        "Launch notebooks from the project directory."
    )


def setup() -> SimpleNamespace:
    """Return a namespace of paths used by every experiment notebook."""
    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)

    root = find_repo_root()
    drive_dataset = root / "PCB_DATA_YOLO"
    drive_runs = root / "runs_pcb"

    if IN_COLAB:
        local_dataset = Path("/content/PCB_DATA_LOCAL")
        local_runs = Path("/content/runs_pcb_local")
        local_noise = Path("/content/PCB_DATA_NOISE")
        local_illum = Path("/content/PCB_DATA_ILLUM")
    else:
        local_dataset = drive_dataset
        local_runs = drive_runs
        local_noise = root / "PCB_DATA_NOISE"
        local_illum = root / "PCB_DATA_ILLUM"

    return SimpleNamespace(
        in_colab=IN_COLAB,
        repo_root=root,
        drive_dataset=drive_dataset,
        drive_data_yaml=drive_dataset / "data.yaml",
        drive_runs=drive_runs,
        drive_cfg=root / "cfg",
        drive_abl=root / "ablation_outputs",
        drive_illum=root / "illumination_outputs",
        report_dir=root / "report_outputs",
        local_dataset=local_dataset,
        local_data_yaml=local_dataset / "data.yaml",
        local_runs=local_runs,
        local_noise=local_noise,
        local_illum=local_illum,
        baseline_best=drive_runs / "yolo12_pcb_baseline" / "weights" / "best.pt",
        p2_cfg=root / "cfg" / "yolo12n_p2.yaml",
        p2_best=drive_runs / "yolo12n_p2_strong" / "weights" / "best.pt",
    )


def prepare_dataset(paths: SimpleNamespace, overwrite: bool = True):
    """Make ``paths.local_dataset`` ready and rewrite ``data.yaml`` path.

    On Colab the Drive copy is cached onto the VM SSD. Locally the
    gitignored ``PCB_DATA_YOLO/`` folder is used in place.
    """
    import yaml

    if paths.in_colab:
        if not paths.drive_data_yaml.exists():
            raise FileNotFoundError(
                f"data.yaml not found on Drive: {paths.drive_data_yaml}"
            )
        if overwrite and paths.local_dataset.exists():
            shutil.rmtree(paths.local_dataset)
        if not paths.local_dataset.exists():
            shutil.copytree(paths.drive_dataset, paths.local_dataset)
    elif not paths.local_data_yaml.exists():
        raise FileNotFoundError(
            f"YOLO dataset missing: {paths.local_data_yaml}\n"
            "Download PCB_DATA_YOLO/ (see README) or run XmlToYolo.ipynb."
        )

    with open(paths.local_data_yaml, "r", encoding="utf-8") as fh:
        yml = yaml.safe_load(fh)
    yml["path"] = str(paths.local_dataset)
    yml["train"] = "images/train"
    yml["val"] = "images/val"
    yml["test"] = "images/test"
    with open(paths.local_data_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(yml, fh, sort_keys=False)
    return yml


def sync_run(src: Path, dst: Path) -> Path:
    """Copy a training run back to Drive. No-op when src and dst are the same."""
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        print("Run directory is already in the repo; skip Drive sync.")
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print("Synced to:", dst)
    return dst
