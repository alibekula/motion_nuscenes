from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_leaderboard import _all_stage_local_trajectories, _anchor_bank_from_sources, _build_model_cfg
from motion_v1.dataloader import V1ArtifactDataset, collate_v1_batch
from motion_v1.model import V1MotionModel


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _select_agent_index(sample: dict[str, Any], agent_token: str | None, agent_index: int | None) -> int:
    tokens = list(sample["agent_tokens"])
    if agent_token is not None:
        if agent_token not in tokens:
            raise ValueError(f"Agent token {agent_token!r} is not present in sample.")
        return int(tokens.index(agent_token))
    if agent_index is not None:
        if agent_index < 0 or agent_index >= len(tokens):
            raise ValueError(f"agent-index must be in [0, {len(tokens) - 1}].")
        return int(agent_index)

    future = sample["future_positions_ego"].numpy()
    history = sample["history_features"].numpy()
    displacement = np.linalg.norm(future[:, -1, 0:2] - history[:, -1, 0:2], axis=-1)
    if len(tokens) > 1:
        displacement[0] = -1.0
    return int(np.argmax(displacement))


def _local_to_ego(local_xy: np.ndarray, last_xy: np.ndarray, last_yaw: float) -> np.ndarray:
    cos_yaw = float(np.cos(last_yaw))
    sin_yaw = float(np.sin(last_yaw))
    rotation_local_to_ego = np.asarray(
        [
            [cos_yaw, sin_yaw],
            [-sin_yaw, cos_yaw],
        ],
        dtype=np.float32,
    )
    return local_xy @ rotation_local_to_ego + last_xy.reshape(1, 1, 2)


def _top1_prediction_ego(all_local: torch.Tensor, logits: torch.Tensor, history: np.ndarray, agent_idx: int) -> np.ndarray:
    class_idx = int(logits[agent_idx].argmax().item())
    local = all_local[agent_idx, class_idx].detach().cpu().numpy()
    last_xy = history[agent_idx, -1, 0:2]
    last_yaw = float(np.arctan2(history[agent_idx, -1, 3], history[agent_idx, -1, 2]))
    return _local_to_ego(local.reshape(1, local.shape[0], 2), last_xy=last_xy, last_yaw=last_yaw)[0]


def _plot_polylines(ax, sample: dict[str, Any]) -> None:
    point_features = sample["polyline_point_features"].numpy()
    point_mask = sample["polyline_point_mask"].numpy().astype(bool)
    for points, mask in zip(point_features, point_mask, strict=False):
        valid = points[mask, 0:2]
        if valid.shape[0] < 2:
            continue
        ax.plot(valid[:, 0], valid[:, 1], color="#B8BDC7", linewidth=0.7, alpha=0.65, zorder=1)


def _plot_objects(ax, sample: dict[str, Any]) -> None:
    objects = sample["object_features"].numpy()
    if objects.size == 0:
        return
    ax.scatter(objects[:, 0], objects[:, 1], s=8, color="#8E96A3", alpha=0.45, zorder=2)


def _plot_trajectory(ax, xy: np.ndarray, *, color: str, label: str, linewidth: float, alpha: float, zorder: int) -> None:
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=linewidth, alpha=alpha, label=label, zorder=zorder)
    ax.scatter(xy[-1:, 0], xy[-1:, 1], color=color, s=18, alpha=alpha, zorder=zorder + 1)


def _plot_full_scene(ax, sample: dict[str, Any], all_local: torch.Tensor, logits: torch.Tensor, max_agents: int) -> None:
    history_all = sample["history_features"].numpy()
    future_all = sample["future_positions_ego"].numpy()
    tokens = list(sample["agent_tokens"])
    agent_indices = list(range(1, len(tokens)))
    if max_agents > 0:
        displacement = np.linalg.norm(future_all[agent_indices, -1, 0:2] - history_all[agent_indices, -1, 0:2], axis=-1)
        order = np.argsort(displacement)[::-1][:max_agents]
        agent_indices = [agent_indices[int(idx)] for idx in order]

    for plot_idx, agent_idx in enumerate(agent_indices):
        pred = _top1_prediction_ego(all_local, logits, history_all, agent_idx)
        show_labels = plot_idx == 0
        ax.plot(
            history_all[agent_idx, :, 0],
            history_all[agent_idx, :, 1],
            color="#20242A",
            linewidth=1.2,
            alpha=0.55,
            label="agent history" if show_labels else None,
            zorder=4,
        )
        ax.plot(
            future_all[agent_idx, :, 0],
            future_all[agent_idx, :, 1],
            color="#18A058",
            linewidth=1.5,
            alpha=0.65,
            label="ground truth" if show_labels else None,
            zorder=5,
        )
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            color="#D7263D",
            linewidth=1.5,
            alpha=0.65,
            label="top1 prediction" if show_labels else None,
            zorder=6,
        )
        ax.scatter(history_all[agent_idx, -1, 0], history_all[agent_idx, -1, 1], s=13, color="#20242A", alpha=0.65, zorder=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a qualitative V1 motion prediction example.")
    parser.add_argument("--artifact", type=str, required=True, help="Artifact .pt path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument("--output", type=str, default="docs/assets/prediction_example.png")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--agent-index", type=int, default=None)
    parser.add_argument("--agent-token", type=str, default=None)
    parser.add_argument("--all-agents", action="store_true", help="Render top1 predictions for all non-ego agents.")
    parser.add_argument("--max-agents", type=int, default=48, help="Limit full-scene rendering to most-moving agents. Use 0 for all.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--xlim", type=float, default=80.0)
    parser.add_argument("--ylim", type=float, default=80.0)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    device = torch.device(args.device)
    dataset = V1ArtifactDataset(args.artifact, augmentation=None)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise ValueError(f"sample-index must be in [0, {len(dataset) - 1}].")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    anchor_bank = _anchor_bank_from_sources(dataset, checkpoint)
    model_cfg = _build_model_cfg(dataset, checkpoint, anchor_bank)
    model = V1MotionModel(model_cfg, anchor_bank)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    sample = dataset[args.sample_index]
    agent_idx = _select_agent_index(sample, args.agent_token, args.agent_index)
    batch = _move_batch_to_device(collate_v1_batch([sample]), device)

    with torch.no_grad():
        outputs = model(batch, gt_cond_weight=0.0)
        stage2 = outputs["stage2"]
        logits_all = stage2["logits"][0].detach().cpu()
        all_local_all = _all_stage_local_trajectories(stage2)[0].detach().cpu()
        probs = torch.softmax(stage2["logits"][0, agent_idx], dim=-1)
        all_local = _all_stage_local_trajectories(stage2)[0, agent_idx]
        topk = min(int(args.topk), int(all_local.shape[0]))
        top_prob, top_idx = probs.topk(topk, dim=-1)
        pred_local = all_local[top_idx].detach().cpu().numpy()
        top_prob_np = top_prob.detach().cpu().numpy()
        top_idx_np = top_idx.detach().cpu().numpy()

    history = sample["history_features"][agent_idx].numpy()
    future = sample["future_positions_ego"][agent_idx].numpy()
    last_xy = history[-1, 0:2]
    last_yaw = float(np.arctan2(history[-1, 3], history[-1, 2]))
    pred_ego = _local_to_ego(pred_local, last_xy=last_xy, last_yaw=last_yaw)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
    ax.set_facecolor("#F7F8FA")
    _plot_polylines(ax, sample)
    _plot_objects(ax, sample)

    ego_history = sample["history_features"][0].numpy()[:, 0:2]
    ax.plot(ego_history[:, 0], ego_history[:, 1], color="#2F6FED", linewidth=1.6, alpha=0.75, label="ego history", zorder=4)

    token = sample["agent_tokens"][agent_idx]
    mode_text = ", ".join(f"{int(idx)}:{prob:.2f}" for idx, prob in zip(top_idx_np, top_prob_np, strict=False))
    if args.all_agents:
        _plot_full_scene(ax, sample, all_local_all, logits_all, max_agents=int(args.max_agents))
        title = f"sample {args.sample_index} | full scene top1 predictions | agents={len(sample['agent_tokens']) - 1}"
    else:
        _plot_trajectory(ax, history[:, 0:2], color="#20242A", label="agent history", linewidth=2.2, alpha=1.0, zorder=6)
        _plot_trajectory(ax, future[:, 0:2], color="#18A058", label="ground truth", linewidth=2.4, alpha=0.95, zorder=7)

        for rank, trajectory in reversed(list(enumerate(pred_ego))):
            is_top1 = rank == 0
            color = "#D7263D" if is_top1 else "#F29E4C"
            label = f"prediction top{rank + 1}" if is_top1 else None
            linewidth = 2.4 if is_top1 else 1.4
            alpha = 0.9 if is_top1 else max(0.18, 0.5 - 0.05 * rank)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=linewidth, alpha=alpha, label=label, zorder=8 if is_top1 else 5)

        ax.scatter([last_xy[0]], [last_xy[1]], marker="o", s=42, color="#20242A", edgecolor="white", linewidth=0.8, zorder=10)
        title = f"sample {args.sample_index} | agent {agent_idx} ({token}) | top modes {mode_text}"

    ax.scatter([0.0], [0.0], marker="x", s=55, color="#2F6FED", linewidths=1.8, label="reference ego", zorder=9)

    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel("x in ego frame, m")
    ax.set_ylabel("y in ego frame, m")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-float(args.xlim), float(args.xlim))
    ax.set_ylim(-float(args.ylim), float(args.ylim))
    ax.grid(True, color="#E2E5EA", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Saved visualization: {output_path}")
    print(f"scene_token={sample['scene_token']} sample_token={sample['sample_token']} agent_token={token}")


if __name__ == "__main__":
    main()
