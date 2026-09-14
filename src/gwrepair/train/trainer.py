import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from gwrepair.utils import ReadConfigs
from gwrepair.train.utils import stft_module
from gwrepair.train.models.attention import AttentionUNet2D
from gwrepair.train.utils.plotting import plot_stft_mag, plot_stft_mask


@dataclass
class TrainConfig:
    type_: str = "waveform"
    num_attention_gates: int = 3
    lr: float = 2e-4
    epochs: int = 50
    grad_clip: float = 1.0
    amp: bool = True
    ckpt_save_epoch: int = 5
    outdir: Path = "./outdir/waveform"
    ckpt: str | None = None
    device: str = "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Attention U-Net for GWRepair masks."
    )

    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--type", dest="type_", type=str, default="waveform")
    parser.add_argument("--num_attention_gates", type=int, default=3)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--ckpt_save_epoch", type=int, default=10)
    parser.add_argument("--outdir", type=Path, default="./outdir/waveform")
    parser.add_argument("--ckpt", type=Path, default=None)

    parser.add_argument("--n_fft", type=int, default=256)
    parser.add_argument("--hop", type=int, default=16)
    parser.add_argument("--win", type=int, default=256)

    return parser.parse_args()


def compute_loss(pred, injection, stft_X, eps: float = 1e-8):
    obs = stft_X[:, :1]

    residual = ((pred.float() - injection.float()) * obs).abs().sum(
        dim=(1, 2, 3)
    )

    norm = (injection.float() * obs).abs().sum(dim=(1, 2, 3)) + eps

    return (residual / norm).mean() * 100


def run_validation(stft_X, stft_mask, model, cfg, device, epoch=-1):
    model.eval()

    stft_X = stft_X.to(device)
    stft_mask = stft_mask.to(device)

    with torch.no_grad():
        with torch.amp.autocast(
            device_type="cuda",
            enabled=(cfg.amp and device.type == "cuda"),
        ):
            pred = model(stft_X)
            loss = compute_loss(pred, stft_mask, stft_X)

        if epoch >= 0 and epoch % 5 == 0:
            plot_dir = Path(cfg.outdir) / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            data_example = stft_X[0, 0].detach().cpu()
            mask_example = stft_mask[0, 0].detach().cpu()
            pred_example = (pred[0, 0].detach().cpu() > 0.5).float()

            fig, ax = plt.subplots(1, figsize=(5, 10))
            plot_stft_mag(ax, data_example)
            plot_stft_mask(ax, mask_example, c="g")
            plot_stft_mask(ax, pred_example, c="C1")

            fig.savefig(plot_dir / f"{epoch}.png", bbox_inches="tight")
            plt.close(fig)

    return loss.detach().item()


def save_checkpoint(
    path,
    epoch,
    model,
    opt,
    train_history,
    train_history_step,
    val_history,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "train_loss": train_history,
        "train_history_step": train_history_step,
        "val_loss": val_history,
    }

    torch.save(checkpoint, path)

def cleanup_old_checkpoints(outdir: Path, keep_last: int = 4):
    ckpts = sorted(
        outdir.glob("epoch_*.ckpt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    for ckpt in ckpts[:-keep_last]:
        ckpt.unlink()


def train(dataset, spec, device, cfg):
    train_loader = dataset.train_dataloader()
    device = torch.device(device)

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    stft_X_val, stft_mask_val = spec(
        dataset.val_batch,
        step="validate",
        injection=cfg.type_,
    )

    model = AttentionUNet2D(
        num_attention_gates=cfg.num_attention_gates
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(cfg.amp and device.type == "cuda"),
    )

    start_epoch = 0
    train_history = []
    val_history = []
    train_history_step = []

    if cfg.ckpt is not None:
        checkpoint = torch.load(cfg.ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        train_history = checkpoint.get("train_loss", [])
        train_history_step = checkpoint.get("train_history_step", [])
        val_history = checkpoint.get("val_loss", [])

        print(f"Loaded checkpoint: {cfg.ckpt}")
        print(f"Resuming from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        batch_losses = []

        for i, batch in enumerate(train_loader):
            batch = batch[0].to(device)

            stft_X, stft_mask = spec(batch, injection=cfg.type_)
            stft_X = stft_X.to(device)
            stft_mask = stft_mask.to(device)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(cfg.amp and device.type == "cuda"),
            ):
                pred = model(stft_X)
                loss = compute_loss(pred, stft_mask, stft_X)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.grad_clip,
            )

            scaler.step(opt)
            scaler.update()

            batch_losses.append(loss.detach().item())

        train_epoch_loss = sum(batch_losses) / len(batch_losses)

        val_epoch_loss = run_validation(
            stft_X_val,
            stft_mask_val,
            model=model,
            cfg=cfg,
            device=device,
            epoch=epoch,
        )

        train_history.append(train_epoch_loss)
        train_history_step.append(batch_losses)
        val_history.append(val_epoch_loss)

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"train loss: {train_epoch_loss:.6f} | "
            f"val loss: {val_epoch_loss:.6f}"
        )

        if (epoch + 1) % cfg.ckpt_save_epoch == 0:
            save_checkpoint(
                outdir / f"epoch_{epoch + 1}.ckpt",
                epoch,
                model,
                opt,
                train_history,
                train_history_step,
                val_history,
            )
            cleanup_old_checkpoints(outdir, keep_last=4)

    save_checkpoint(
        outdir / "last.ckpt",
        cfg.epochs - 1,
        model,
        opt,
        train_history,
        train_history_step,
        val_history,
    )


def main():
    args = parse_args()

    cfg = TrainConfig(
        type_=args.type_,
        num_attention_gates=args.num_attention_gates,
        lr=args.lr,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        amp=bool(args.amp),
        ckpt_save_epoch=args.ckpt_save_epoch,
        outdir=args.outdir,
        ckpt=args.ckpt,
        device=args.device,
    )

    dataset = ReadConfigs(args.root)
    dataset.device = cfg.device

    ## information ##
    text = f"training {cfg.type_} model"
    print(text)
    text = f"Model: Attention U-Net 2D, num_att_gates: {cfg.num_attention_gates}"
    print(text)
    text = f"Trainging: total epochs: {cfg.epochs}, batch_size: {dataset.batch_size}, batches_per_epoch: {dataset.batches_per_epoch}"
    print(text)
    text = f"waveform snr range: {dataset.waveform_snr}, glitch snr range: {dataset.glitch_snr}"
    print(text)
    text = f"Output directory: {cfg.outdir.resolve()}"
    print(text)

    dataset.setup(injection=cfg.type_)
    spec = stft_module(
        n_fft=args.n_fft,
        hop=args.hop,
        win=args.win,
        data_module=dataset,
    ).to(cfg.device)

    train(
        dataset=dataset,
        spec=spec,
        device=args.device,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
