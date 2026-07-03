from pathlib import Path
import os
import logging
import random
import csv
import shutil
from datetime import datetime

def setup_logger(outdir):

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(outdir / "training.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def _read_keep_flag(meta_file: Path) -> bool:
    """
    Read keep flag from meta.txt.
    Expected line format: keep: true / keep: false
    """
    if not meta_file.exists():
        return False

    for line in meta_file.read_text().splitlines():
        if line.lower().startswith("keep:"):
            value = line.split(":", 1)[1].strip().lower()
            return value in {"true", "yes", "1", "keep"}
    return False


def _write_meta(meta_file: Path, keep: bool):
    """
    Write metadata for the current run.
    """
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_file.write_text(
        f"created_at: {created_at}\n"
        f"keep: {str(keep).lower()}\n"
    )


def prepare_train_dir(outdir: str | Path, keep: bool, name: str = "train"):
    """
    Prepare a fresh training directory.

    Behavior:
    - If `outdir/name` exists and its meta.txt says keep: true:
        rename it to `name_<random3digits>`
    - Otherwise:
        delete it
    - Then create a fresh `outdir/name`
    - Write meta.txt with created_at and keep flag
    """
    outdir = Path(outdir)
    train_dir = outdir / name
    meta_file = train_dir / "meta.txt"

    if train_dir.exists():
        prev_keep = _read_keep_flag(meta_file)

        if prev_keep:
            while True:
                archived = outdir / f"{name}_{random.randint(100, 999)}"
                if not archived.exists():
                    train_dir.rename(archived)
                    break
        else:
            shutil.rmtree(train_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    _write_meta(train_dir / "meta.txt", keep)
    return train_dir, setup_logger(train_dir)


def log_metrics_csv(filepath, epoch, train_metrics, val_metrics):
    file_exists = os.path.exists(filepath)

    row = {"epoch": epoch}
    train_metrics = {"train_" + k: train_metrics[k] for k in train_metrics.keys()}
    val_metrics = {"val_" + k: val_metrics[k] for k in val_metrics.keys()}
    row.update(train_metrics)
    row.update(val_metrics)
    
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)