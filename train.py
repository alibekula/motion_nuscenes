from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from motion_v1.dataloader import V1ArtifactDataset, V1AugmentationConfig, build_v1_loader
from motion_v1.model import V1LossConfig, V1ModelConfig, V1MotionModel, compute_v1_losses


def _meta_get(metadata, key):
    if isinstance(metadata, dict):
        return metadata[key]
    return getattr(metadata, key)


def move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_cfg,
    scaler,
    grad_clip=1.0,
    use_amp=True,
):
    model.train()
    running = {}
    n_batches = 0
    amp_enabled = bool(use_amp and device.type == "cuda")

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
        with amp_ctx:
            outputs = model(batch, gt_cond_weight=0.0)
            loss, metrics = compute_v1_losses(outputs, batch, loss_cfg)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=False)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=False)
            optimizer.step()

        metrics["grad_norm"] = float(grad_norm.item()) if torch.isfinite(grad_norm) else 0.0
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])

        for key, value in metrics.items():
            running[key] = running.get(key, 0.0) + float(value)

        n_batches += 1
        pbar.set_postfix(
            loss=f"{running['total'] / n_batches:.4f}",
            ade=f"{running['stage2_ade'] / n_batches:.4f}",
            lr=f"{metrics['lr']:.2e}",
        )

    return {key: value / max(n_batches, 1) for key, value in running.items()}


@torch.no_grad()
def validate_one_epoch(model, loader, device, loss_cfg):
    model.eval()
    running = {}
    n_batches = 0

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch, gt_cond_weight=0.0)
        _, metrics = compute_v1_losses(outputs, batch, loss_cfg)

        for key, value in metrics.items():
            running[key] = running.get(key, 0.0) + float(value)
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in running.items()}


def main():
    parser = argparse.ArgumentParser(description="Обучение Motion Prediction V1")
    parser.add_argument("--train-artifact", type=str, required=True, help="Путь к train-артефакту .pt")
    parser.add_argument("--val-artifact", type=str, required=True, help="Путь к val-артефакту .pt")
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_aug = V1AugmentationConfig(
        enabled=True,
        apply_prob=0.8,
        max_rotation_deg=10.0,
        translation_std_m=0.0,
    )

    train_dataset = V1ArtifactDataset(args.train_artifact, augmentation=train_aug)
    val_dataset = V1ArtifactDataset(args.val_artifact, augmentation=None)

    train_loader = build_v1_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_v1_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metadata = train_dataset.metadata
    anchor_bank = train_dataset.anchor_bank
    if anchor_bank is None:
        raise ValueError("Train artifact must contain an anchor bank.")

    model_cfg = V1ModelConfig(
        history_feature_dim=int(_meta_get(metadata, "history_feature_dim")),
        polyline_point_dim=int(_meta_get(metadata, "polyline_point_dim")),
        polyline_attr_dim=int(_meta_get(metadata, "polyline_attr_dim")),
        object_feature_dim=int(_meta_get(metadata, "object_feature_dim")),
        k12=int(len(anchor_bank["full_bank"])),
        k6=int(len(anchor_bank["prefix_bank"])),
        r_max=float(anchor_bank.get("r_max", 1.0)),
        dropout=0.2,
    )
    model = V1MotionModel(model_cfg, anchor_bank).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    loss_cfg = V1LossConfig(
        cls_focal_gamma=1.5,
        stationary_cls_weight=0.4,
        soft_anchor_tau=0.2,
        soft_anchor_topk=6,
        regression_mode="predicted_topk",
        predicted_anchor_topk=3,
        predicted_anchor_detach=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.disable_amp))
    best_val_ade = float("inf")
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_cfg,
            scaler,
            grad_clip=args.grad_clip,
            use_amp=not args.disable_amp,
        )
        val_metrics = validate_one_epoch(model, val_loader, device, loss_cfg)
        scheduler.step()

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Train ADE: {train_metrics['stage2_ade']:.4f} | "
            f"Val ADE: {val_metrics['stage2_ade']:.4f} | "
            f"Val FDE: {val_metrics['stage2_fde_l2']:.4f}"
        )

        if val_metrics["stage2_ade"] < best_val_ade:
            best_val_ade = val_metrics["stage2_ade"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_cfg": model_cfg,
                    "loss_cfg": loss_cfg,
                    "best_val_ade": best_val_ade,
                },
                checkpoint_path,
            )
            print(f"  -> Сохранён новый лучший чекпоинт: {checkpoint_path}")


if __name__ == "__main__":
    main()
