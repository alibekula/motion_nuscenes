from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class PreprocessedDatasetMetadata:
    split_name: str
    history_frames: int
    future_frames: int
    history_feature_dim: int
    polyline_point_dim: int
    polyline_attr_dim: int
    object_feature_dim: int
    edge_attr_dim: int


class PreprocessedMotionDataset(Dataset):
    """
    RAM-backed dataset over offline-preprocessed motion prediction samples.

    Expected artifact layout:
        {
            "metadata": {...},
            "samples": [ sample_dict, ... ],
            "anchor_bank": optional_dict,
            "coverage_report": optional_dict,
        }
    """

    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact_path = str(artifact_path)
        payload = torch.load(self.artifact_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("Expected preprocessed artifact payload to be a dict.")
        if "samples" not in payload or "metadata" not in payload:
            raise ValueError("Artifact must contain 'samples' and 'metadata'.")

        metadata = payload["metadata"]
        if isinstance(metadata, PreprocessedDatasetMetadata):
            self.metadata = metadata
        elif isinstance(metadata, dict):
            self.metadata = PreprocessedDatasetMetadata(**metadata)
        else:
            raise ValueError("metadata must be a dict or PreprocessedDatasetMetadata.")

        samples = payload["samples"]
        if not isinstance(samples, list):
            raise ValueError("samples must be stored as a list.")
        self.samples = samples
        self.anchor_bank = payload.get("anchor_bank")
        self.coverage_report = payload.get("coverage_report")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        if not isinstance(item, dict):
            raise ValueError("Each preprocessed sample must be a dict.")
        return item


def collate_preprocessed_motion_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_agents = max((int(item["num_agents"]) for item in batch), default=0)
    max_polylines = max((int(item["num_polylines"]) for item in batch), default=0)
    max_points = max((int(item["num_polyline_points"]) for item in batch), default=0)
    max_objects = max((int(item["num_objects"]) for item in batch), default=0)
    max_edges = max((int(item["num_edges"]) for item in batch), default=0)

    history_frames = batch[0]["history_features"].shape[1] if batch else 0
    future_frames = batch[0]["future_positions_ego"].shape[1] if batch else 0
    history_feature_dim = batch[0]["history_features"].shape[2] if batch else 0
    polyline_point_dim = batch[0]["polyline_point_features"].shape[2] if batch and batch[0]["num_polylines"] > 0 else 4
    polyline_attr_dim = batch[0]["polyline_attrs"].shape[1] if batch and batch[0]["num_polylines"] > 0 else 8
    object_feature_dim = batch[0]["object_features"].shape[1] if batch and batch[0]["num_objects"] > 0 else 10
    edge_attr_dim = batch[0]["object_polyline_edge_attr"].shape[1] if batch and batch[0]["num_edges"] > 0 else 12

    def zeros(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    packed: dict[str, Any] = {
        "class_ids": torch.full((batch_size, max_agents), -1, dtype=torch.long),
        "history_features": zeros(batch_size, max_agents, history_frames, history_feature_dim, dtype=torch.float32),
        "history_positions_ego": zeros(batch_size, max_agents, history_frames, 2, dtype=torch.float32),
        "history_yaw_ego": zeros(batch_size, max_agents, history_frames, dtype=torch.float32),
        "history_vel_xy": zeros(batch_size, max_agents, history_frames, 2, dtype=torch.float32),
        "history_yaw_rate": zeros(batch_size, max_agents, history_frames, dtype=torch.float32),
        "history_valid_mask": zeros(batch_size, max_agents, history_frames, dtype=torch.bool),
        "history_missing_mask": zeros(batch_size, max_agents, history_frames, dtype=torch.bool),
        "history_len": zeros(batch_size, max_agents, dtype=torch.long),
        "future_positions_ego": zeros(batch_size, max_agents, future_frames, 2, dtype=torch.float32),
        "future_yaw_ego": zeros(batch_size, max_agents, future_frames, dtype=torch.float32),
        "future_valid_mask": zeros(batch_size, max_agents, future_frames, dtype=torch.bool),
        "future_missing_mask": zeros(batch_size, max_agents, future_frames, dtype=torch.bool),
        "future_len": zeros(batch_size, max_agents, dtype=torch.long),
        "future_gt_len": zeros(batch_size, max_agents, dtype=torch.long),
        "current_xy_ego": zeros(batch_size, max_agents, 2, dtype=torch.float32),
        "p_last_hist_ego": zeros(batch_size, max_agents, 2, dtype=torch.float32),
        "current_yaw_ego": zeros(batch_size, max_agents, dtype=torch.float32),
        "agent_yaw_ego": zeros(batch_size, max_agents, dtype=torch.float32),
        "current_size_xy": zeros(batch_size, max_agents, 2, dtype=torch.float32),
        "current_speed": zeros(batch_size, max_agents, dtype=torch.float32),
        "rotation_local_to_ego": zeros(batch_size, max_agents, 2, 2, dtype=torch.float32),
        "rotation_ego_to_local": zeros(batch_size, max_agents, 2, 2, dtype=torch.float32),
        "train_valid_mask": zeros(batch_size, max_agents, dtype=torch.bool),
        "inference_valid_mask": zeros(batch_size, max_agents, dtype=torch.bool),
        "agent_pad_mask": torch.ones((batch_size, max_agents), dtype=torch.bool),
        "polyline_point_features": zeros(batch_size, max_polylines, max_points, polyline_point_dim, dtype=torch.float32),
        "polyline_point_mask": zeros(batch_size, max_polylines, max_points, dtype=torch.bool),
        "polyline_attrs": zeros(batch_size, max_polylines, polyline_attr_dim, dtype=torch.float32),
        "polyline_pad_mask": torch.ones((batch_size, max_polylines), dtype=torch.bool),
        "object_features": zeros(batch_size, max_objects, object_feature_dim, dtype=torch.float32),
        "object_pad_mask": torch.ones((batch_size, max_objects), dtype=torch.bool),
        "object_polyline_edge_index": zeros(batch_size, max_edges, 2, dtype=torch.long),
        "object_polyline_edge_attr": zeros(batch_size, max_edges, edge_attr_dim, dtype=torch.float32),
        "object_polyline_edge_mask": zeros(batch_size, max_edges, dtype=torch.bool),
    }

    sample_tokens: list[str] = []
    scene_tokens: list[str] = []
    agent_tokens: list[list[str]] = []
    map_names: list[str] = []

    for batch_idx, item in enumerate(batch):
        num_agents = int(item["num_agents"])
        num_polylines = int(item["num_polylines"])
        num_objects = int(item["num_objects"])
        num_edges = int(item["num_edges"])
        num_points = int(item["num_polyline_points"])

        sample_tokens.append(str(item["sample_token"]))
        scene_tokens.append(str(item["scene_token"]))
        agent_tokens.append(list(item["agent_tokens"]))
        map_names.append(str(item.get("map_name", "")))

        if num_agents > 0:
            packed["agent_pad_mask"][batch_idx, :num_agents] = False
            for name in (
                "class_ids",
                "history_features",
                "history_positions_ego",
                "history_yaw_ego",
                "history_vel_xy",
                "history_yaw_rate",
                "history_valid_mask",
                "history_missing_mask",
                "history_len",
                "future_positions_ego",
                "future_yaw_ego",
                "future_valid_mask",
                "future_missing_mask",
                "future_len",
                "future_gt_len",
                "current_xy_ego",
                "p_last_hist_ego",
                "current_yaw_ego",
                "agent_yaw_ego",
                "current_size_xy",
                "current_speed",
                "rotation_local_to_ego",
                "rotation_ego_to_local",
                "train_valid_mask",
                "inference_valid_mask",
            ):
                packed[name][batch_idx, :num_agents] = item[name]

        if num_polylines > 0:
            packed["polyline_pad_mask"][batch_idx, :num_polylines] = False
            packed["polyline_point_features"][batch_idx, :num_polylines, :num_points] = item["polyline_point_features"]
            packed["polyline_point_mask"][batch_idx, :num_polylines, :num_points] = item["polyline_point_mask"]
            packed["polyline_attrs"][batch_idx, :num_polylines] = item["polyline_attrs"]

        if num_objects > 0:
            packed["object_pad_mask"][batch_idx, :num_objects] = False
            packed["object_features"][batch_idx, :num_objects] = item["object_features"]

        if num_edges > 0:
            packed["object_polyline_edge_mask"][batch_idx, :num_edges] = True
            packed["object_polyline_edge_index"][batch_idx, :num_edges] = item["object_polyline_edge_index"]
            packed["object_polyline_edge_attr"][batch_idx, :num_edges] = item["object_polyline_edge_attr"]

    packed["sample_tokens"] = sample_tokens
    packed["scene_tokens"] = scene_tokens
    packed["agent_tokens"] = agent_tokens
    packed["map_names"] = map_names
    return packed


def build_preprocessed_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_preprocessed_motion_batch,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)
