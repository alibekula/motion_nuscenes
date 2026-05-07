from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from motion_v1.dataloader import V1ArtifactDataset, build_v1_loader
from motion_v1.model import V1ModelConfig, V1MotionModel, future_positions_local_from_history


LEADERBOARD_METRICS = (
    "MinADE_5",
    "MinADE_10",
    "MissRateTopK_2_5",
    "MissRateTopK_2_10",
    "MinFDE_1",
)


def _meta_get(metadata: Any, key: str) -> Any:
    if isinstance(metadata, dict):
        return metadata[key]
    return getattr(metadata, key)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _anchor_bank_from_sources(dataset: V1ArtifactDataset, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if dataset.anchor_bank is not None:
        return dataset.anchor_bank

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict) or "A12" not in state_dict or "A6" not in state_dict:
        raise ValueError("Artifact has no anchor bank and checkpoint does not contain A12/A6 buffers.")

    model_cfg = checkpoint.get("model_cfg")
    return {
        "full_bank": state_dict["A12"].detach().cpu(),
        "prefix_bank": state_dict["A6"].detach().cpu(),
        "r_max": float(getattr(model_cfg, "r_max", 1.0)),
        "method": "checkpoint_buffers",
    }


def _build_model_cfg(dataset: V1ArtifactDataset, checkpoint: dict[str, Any], anchor_bank: dict[str, Any]) -> V1ModelConfig:
    if isinstance(checkpoint.get("model_cfg"), V1ModelConfig):
        return checkpoint["model_cfg"]

    metadata = dataset.metadata
    train_args = checkpoint.get("train_args", {})
    return V1ModelConfig(
        history_feature_dim=int(_meta_get(metadata, "history_feature_dim")),
        polyline_point_dim=int(_meta_get(metadata, "polyline_point_dim")),
        polyline_attr_dim=int(_meta_get(metadata, "polyline_attr_dim")),
        object_feature_dim=int(_meta_get(metadata, "object_feature_dim")),
        k12=int(len(anchor_bank["full_bank"])),
        k6=int(len(anchor_bank["prefix_bank"])),
        r_max=float(anchor_bank.get("r_max", 1.0)),
        dropout=float(train_args.get("dropout", 0.0)),
        use_pose_encoding=not bool(train_args.get("disable_pose_encoding", False)),
        use_relative_encoding=not bool(train_args.get("disable_relative_encoding", False)),
        pose_radius_m=float(train_args.get("pose_radius_m", 80.0)),
    )


def _all_stage_local_trajectories(stage_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    directional = stage_outputs["directional_local_trajectories"]
    zero = directional.new_zeros(
        directional.shape[0],
        directional.shape[1],
        1,
        directional.shape[3],
        directional.shape[4],
    )
    return torch.cat([directional, zero], dim=2)


def _ego_mask_from_tokens(batch: dict[str, Any], valid_mask: torch.Tensor) -> torch.Tensor:
    ego_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    for batch_idx, tokens in enumerate(batch.get("agent_tokens", [])):
        for agent_idx, token in enumerate(tokens):
            if agent_idx < ego_mask.shape[1] and str(token) == "ego":
                ego_mask[batch_idx, agent_idx] = True
    return ego_mask


@torch.no_grad()
def evaluate_leaderboard_like(
    model: V1MotionModel,
    loader,
    device: torch.device,
    *,
    exclude_ego: bool,
    miss_threshold_m: float,
) -> dict[str, float | int | None]:
    model.eval()

    sums = {
        "Top1ADE": 0.0,
        "MinADE_5": 0.0,
        "MinADE_10": 0.0,
        "MinFDE_1": 0.0,
        "MissRateTopK_2_5": 0.0,
        "MissRateTopK_2_10": 0.0,
    }
    total_agents = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        batch = _move_batch_to_device(batch, device)
        outputs = model(batch, gt_cond_weight=0.0)

        stage2 = outputs["stage2"]
        logits = stage2["logits"]
        local = _all_stage_local_trajectories(stage2)
        horizon = min(int(local.shape[-2]), int(batch["future_positions_ego"].shape[-2]))

        target = future_positions_local_from_history(batch)[..., :horizon, :]
        local = local[..., :horizon, :]

        valid_mask = ~batch["agent_pad_mask"].to(dtype=torch.bool)
        if exclude_ego:
            valid_mask = valid_mask & ~_ego_mask_from_tokens(batch, valid_mask)
        if not torch.any(valid_mask):
            continue

        max_k = min(10, int(logits.shape[-1]))
        top_idx = logits.argsort(dim=-1, descending=True)[..., :max_k]
        gather_idx = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, horizon, 2)
        top_local = local.gather(dim=2, index=gather_idx)

        l2 = torch.linalg.norm(top_local - target.unsqueeze(2), dim=-1)
        ade_per_mode = l2.mean(dim=-1)
        fde_per_mode = torch.linalg.norm(top_local[..., -1, :] - target.unsqueeze(2)[..., -1, :], dim=-1)

        agent_count = int(valid_mask.sum().item())
        total_agents += agent_count

        top1_ade = ade_per_mode[..., 0]
        top1_fde = fde_per_mode[..., 0]
        sums["Top1ADE"] += float(top1_ade[valid_mask].sum().item())
        sums["MinFDE_1"] += float(top1_fde[valid_mask].sum().item())

        for k in (5, 10):
            actual_k = min(k, max_k)
            min_ade = ade_per_mode[..., :actual_k].min(dim=-1).values
            min_fde = fde_per_mode[..., :actual_k].min(dim=-1).values
            sums[f"MinADE_{k}"] += float(min_ade[valid_mask].sum().item())
            sums[f"MissRateTopK_2_{k}"] += float((min_fde[valid_mask] > miss_threshold_m).float().sum().item())

    if total_agents <= 0:
        raise ValueError("No valid agents were evaluated.")

    metrics: dict[str, float | int | None] = {
        "num_eval_agents": total_agents,
        "exclude_ego": int(exclude_ego),
        "miss_threshold_m": float(miss_threshold_m),
        "Top1ADE": sums["Top1ADE"] / total_agents,
        "MinADE_5": sums["MinADE_5"] / total_agents,
        "MinADE_10": sums["MinADE_10"] / total_agents,
        "MinFDE_1": sums["MinFDE_1"] / total_agents,
        "MissRateTopK_2_5": sums["MissRateTopK_2_5"] / total_agents,
        "MissRateTopK_2_10": sums["MissRateTopK_2_10"] / total_agents,
        "OffRoadRate": None,
    }
    return metrics


def _leaderboard_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in data:
        try:
            meta = item["meta"]
            submit_meta = meta["submit_meta"]
            metrics = item["result"][0]["test_split"]
        except (KeyError, IndexError, TypeError):
            continue
        row = {
            "method": submit_meta.get("method_name", "unknown"),
            "submitted_at": str(submit_meta.get("submitted_at", ""))[:10],
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _print_leaderboard_comparison(metrics: dict[str, float | int | None], leaderboard_json: Path) -> None:
    rows = _leaderboard_rows(leaderboard_json)
    print("\nPseudo-rank against leaderboard.json (not official; your split/artifact differs):")
    for metric in LEADERBOARD_METRICS:
        value = metrics.get(metric)
        official_values = [float(row[metric]) for row in rows if metric in row and row[metric] is not None]
        if value is None or not official_values:
            continue
        value_f = float(value)
        rank = 1 + sum(official_value < value_f for official_value in official_values)
        print(f"  {metric}: {value_f:.4f} -> pseudo rank {rank}/{len(official_values) + 1}")

    if "MinADE_5" in metrics and metrics["MinADE_5"] is not None:
        value = float(metrics["MinADE_5"])
        by_minade5 = sorted(
            (row for row in rows if "MinADE_5" in row),
            key=lambda row: float(row["MinADE_5"]),
        )
        print("\nTop leaderboard rows by MinADE_5:")
        for row in by_minade5[:10]:
            print(
                f"  {float(row['MinADE_5']):.4f} | "
                f"{float(row.get('MinADE_10', float('nan'))):.4f} | "
                f"{float(row.get('MinFDE_1', float('nan'))):.4f} | "
                f"{row['method']} ({row['submitted_at']})"
            )
        print(f"\nYour validation MinADE_5: {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V1 with nuScenes prediction leaderboard-like metrics.")
    parser.add_argument("--artifact", type=str, required=True, help="Validation artifact .pt path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument("--leaderboard-json", type=str, default=None, help="Optional nuScenes leaderboard.json path.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save computed metrics.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-ego", action="store_true", help="Include ego agent in metrics.")
    parser.add_argument("--miss-threshold-m", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = V1ArtifactDataset(args.artifact, augmentation=None)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    anchor_bank = _anchor_bank_from_sources(dataset, checkpoint)
    model_cfg = _build_model_cfg(dataset, checkpoint, anchor_bank)
    model = V1MotionModel(model_cfg, anchor_bank)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loader = build_v1_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        generator=None,
    )

    metrics = evaluate_leaderboard_like(
        model,
        loader,
        device,
        exclude_ego=not args.include_ego,
        miss_threshold_m=float(args.miss_threshold_m),
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("\nNotes:")
    print("  - These are leaderboard-like validation metrics, not official nuScenes test-server scores.")
    print("  - OffRoadRate is not computed because this script does not call the official map evaluator.")
    print("  - By default ego is excluded; pass --include-ego to match the current train.py aggregate.")

    if args.leaderboard_json is not None:
        _print_leaderboard_comparison(metrics, Path(args.leaderboard_json))

    if args.output_json is not None:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
