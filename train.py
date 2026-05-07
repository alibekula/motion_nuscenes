from __future__ import annotations

import argparse
import hashlib
import random
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
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


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _positive_int_or_none(value):
    value = int(value)
    return value if value > 0 else None


def _loss_cfg_for_epoch(base_cfg, epoch_idx, stage1_warmup_epochs):
    if epoch_idx < int(stage1_warmup_epochs):
        return replace(base_cfg, stage2_weight=0.0)
    return base_cfg


def _set_module_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad_(bool(trainable))


def _create_ema_state(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if torch.is_floating_point(value)
    }


@torch.no_grad()
def _update_ema_state(model, ema_state, decay):
    model_state = model.state_dict()
    for name, ema_value in ema_state.items():
        ema_value.mul_(float(decay)).add_(model_state[name].detach(), alpha=1.0 - float(decay))


@torch.no_grad()
def _swap_in_ema_state(model, ema_state):
    if ema_state is None:
        return None

    backup = {}
    model_state = model.state_dict()
    for name, ema_value in ema_state.items():
        backup[name] = model_state[name].detach().clone()
        model_state[name].copy_(ema_value)
    return backup


@torch.no_grad()
def _restore_model_state(model, backup):
    if backup is None:
        return

    model_state = model.state_dict()
    for name, value in backup.items():
        model_state[name].copy_(value)


def _state_dict_for_save(model, ema_state=None):
    model_state = model.state_dict()
    out = {}
    for name, value in model_state.items():
        source = ema_state.get(name, value) if ema_state is not None else value
        out[name] = source.detach().cpu().clone()
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
    gt_cond_weight=0.0,
    freeze_decoder1=False,
    ema_state=None,
    ema_decay=0.0,
):
    model.train()
    if freeze_decoder1:
        model.decoder1.eval()
    running = {}
    n_batches = 0
    amp_enabled = bool(use_amp and device.type == "cuda")

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
        with amp_ctx:
            outputs = model(batch, gt_cond_weight=gt_cond_weight)
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

        if ema_state is not None:
            _update_ema_state(model, ema_state, ema_decay)

        metrics["grad_norm"] = float(grad_norm.item()) if torch.isfinite(grad_norm) else 0.0
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])

        for key, value in metrics.items():
            running[key] = running.get(key, 0.0) + float(value)

        n_batches += 1
        pbar.set_postfix(
            loss=f"{running['total'] / n_batches:.4f}",
            top1_ade=f"{running['stage2_top1_ade'] / n_batches:.4f}",
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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.0, help="Use 0 to disable EMA validation/checkpointing.")
    parser.add_argument("--disable-pose-encoding", action="store_true")
    parser.add_argument("--disable-relative-encoding", action="store_true")
    parser.add_argument("--pose-radius-m", type=float, default=80.0)
    parser.add_argument("--train-aug-prob", type=float, default=0.8)
    parser.add_argument("--train-max-rotation-deg", type=float, default=10.0)
    parser.add_argument("--train-translation-std-m", type=float, default=0.0)
    parser.add_argument("--train-history-xy-noise-std-m", type=float, default=0.0)
    parser.add_argument("--train-history-yaw-noise-std-deg", type=float, default=0.0)
    parser.add_argument("--stage1-weight", type=float, default=0.5)
    parser.add_argument("--stage2-weight", type=float, default=1.0)
    parser.add_argument("--stage1-warmup-epochs", type=int, default=0)
    parser.add_argument("--freeze-decoder1-after-warmup", action="store_true")
    parser.add_argument("--cls-focal-gamma", type=float, default=1.5)
    parser.add_argument("--stationary-cls-weight", type=float, default=0.4)
    parser.add_argument("--soft-anchor-tau", type=float, default=0.2)
    parser.add_argument("--soft-anchor-topk", type=int, default=6, help="Use 0 to disable top-k masking.")
    parser.add_argument("--regression-mode", type=str, choices=("gt_wta", "predicted_topk"), default="predicted_topk")
    parser.add_argument("--predicted-anchor-topk", type=int, default=3, help="Use 0 to use all predicted anchors.")
    parser.add_argument("--predicted-anchor-detach", dest="predicted_anchor_detach", action="store_true", default=True)
    parser.add_argument("--no-predicted-anchor-detach", dest="predicted_anchor_detach", action="store_false")
    parser.add_argument("--gt-cond-weight", type=float, default=0.0)
    args = parser.parse_args()

    _seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    train_artifact_hash = _sha256_file(args.train_artifact)
    val_artifact_hash = _sha256_file(args.val_artifact)
    print(f"Train artifact SHA256: {train_artifact_hash}")
    print(f"Val artifact SHA256: {val_artifact_hash}")

    train_aug = V1AugmentationConfig(
        enabled=True,
        apply_prob=args.train_aug_prob,
        max_rotation_deg=args.train_max_rotation_deg,
        translation_std_m=args.train_translation_std_m,
        history_xy_noise_std_m=args.train_history_xy_noise_std_m,
        history_yaw_noise_std_deg=args.train_history_yaw_noise_std_deg,
    )
    print(f"Train augmentation config: {train_aug}")

    train_dataset = V1ArtifactDataset(args.train_artifact, augmentation=train_aug)
    val_dataset = V1ArtifactDataset(args.val_artifact, augmentation=None)
    train_generator = torch.Generator()
    train_generator.manual_seed(int(args.seed))
    val_generator = torch.Generator()
    val_generator.manual_seed(int(args.seed) + 1)

    train_loader = build_v1_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=train_generator,
    )
    val_loader = build_v1_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generator=val_generator,
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
        dropout=args.dropout,
        use_pose_encoding=not args.disable_pose_encoding,
        use_relative_encoding=not args.disable_relative_encoding,
        pose_radius_m=args.pose_radius_m,
    )
    model = V1MotionModel(model_cfg, anchor_bank).to(device)
    if args.ema_decay < 0.0 or args.ema_decay >= 1.0:
        raise ValueError("--ema-decay must be in [0, 1). Use 0 to disable EMA.")
    ema_state = _create_ema_state(model) if args.ema_decay > 0.0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    loss_cfg = V1LossConfig(
        stage1_weight=args.stage1_weight,
        stage2_weight=args.stage2_weight,
        cls_focal_gamma=args.cls_focal_gamma,
        stationary_cls_weight=args.stationary_cls_weight,
        soft_anchor_tau=args.soft_anchor_tau,
        soft_anchor_topk=_positive_int_or_none(args.soft_anchor_topk),
        regression_mode=args.regression_mode,
        predicted_anchor_topk=_positive_int_or_none(args.predicted_anchor_topk),
        predicted_anchor_detach=args.predicted_anchor_detach,
    )
    print(f"Model config: {model_cfg}")
    print(f"Loss config: {loss_cfg}")
    print(f"EMA decay: {args.ema_decay if ema_state is not None else 'off'}")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.disable_amp))
    best_val_top1_ade = float("inf")
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    decoder1_frozen = False

    for epoch in range(args.epochs):
        should_freeze_decoder1 = (
            args.freeze_decoder1_after_warmup
            and int(args.stage1_warmup_epochs) > 0
            and epoch >= int(args.stage1_warmup_epochs)
        )
        if should_freeze_decoder1 and not decoder1_frozen:
            _set_module_trainable(model.decoder1, False)
            decoder1_frozen = True
            print(f"Decoder1 frozen after {args.stage1_warmup_epochs} warmup epochs.")

        train_loss_cfg = _loss_cfg_for_epoch(loss_cfg, epoch, args.stage1_warmup_epochs)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_loss_cfg,
            scaler,
            grad_clip=args.grad_clip,
            use_amp=not args.disable_amp,
            gt_cond_weight=args.gt_cond_weight,
            freeze_decoder1=decoder1_frozen,
            ema_state=ema_state,
            ema_decay=args.ema_decay,
        )
        ema_backup = _swap_in_ema_state(model, ema_state)
        try:
            val_metrics = validate_one_epoch(model, val_loader, device, loss_cfg)
        finally:
            _restore_model_state(model, ema_backup)
        scheduler.step()

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"S1/S2 weight: {train_loss_cfg.stage1_weight:.2f}/{train_loss_cfg.stage2_weight:.2f} | "
            f"D1 frozen: {int(decoder1_frozen)} | "
            f"Train top1 ADE: {train_metrics['stage2_top1_ade']:.4f} | "
            f"Val top1 ADE: {val_metrics['stage2_top1_ade']:.4f} | "
            f"Val top1 FDE: {val_metrics['stage2_top1_fde_l2']:.4f}"
        )

        if val_metrics["stage2_top1_ade"] < best_val_top1_ade:
            best_val_top1_ade = val_metrics["stage2_top1_ade"]
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": _state_dict_for_save(model, ema_state),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_cfg": model_cfg,
                "loss_cfg": loss_cfg,
                "best_val_top1_ade": best_val_top1_ade,
                "ema_decay": float(args.ema_decay) if ema_state is not None else 0.0,
                "validated_with_ema": ema_state is not None,
                "seed": int(args.seed),
                "train_artifact_path": str(Path(args.train_artifact).resolve()),
                "val_artifact_path": str(Path(args.val_artifact).resolve()),
                "train_artifact_sha256": train_artifact_hash,
                "val_artifact_sha256": val_artifact_hash,
                "train_args": vars(args),
            }
            if ema_state is not None:
                checkpoint["raw_model_state_dict"] = _state_dict_for_save(model)
            torch.save(checkpoint, checkpoint_path)
            print(f"  -> Сохранён новый лучший чекпоинт: {checkpoint_path}")


if __name__ == "__main__":
    main()
