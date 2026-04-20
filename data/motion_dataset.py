from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.nuscenes_utils import build_scene_timelines
from motion_v1.categories import AGENT_CLASS_NAMES, EGO_CLASS_NAME, normalize_motion_agent_class
from motion_v1.geometry import (
    global_xy_to_ego,
    interpolate_angle,
    interpolate_xy,
    quaternion_yaw,
    wrap_angle,
    yaw_to_rotation_matrix,
)


@dataclass(frozen=True)
class MotionDatasetConfig:
    history_frames: int = 4
    future_frames: int = 12
    dt: float = 0.5
    min_history_frames: int = 2
    class_names: tuple[str, ...] = AGENT_CLASS_NAMES
    ego_size: tuple[float, float] = (1.9, 4.8)
    include_ego: bool = True
    max_agents: int | None = None
    interpolate_single_frame_gaps: bool = True


@dataclass(frozen=True)
class AgentState:
    sample_token: str
    scene_index: int
    instance_token: str
    class_name: str
    xy_global: np.ndarray
    yaw_global: float
    size_xy: np.ndarray
    missing: bool


class MotionPredictionDataset(Dataset):
    """One item = one keyframe with all movable agents in the scene."""

    def __init__(self, nusc, cfg: MotionDatasetConfig | None = None, sample_tokens: list[str] | None = None) -> None:
        self.nusc = nusc
        self.cfg = cfg or MotionDatasetConfig()
        self.class_names = tuple(self.cfg.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.history_frames = int(self.cfg.history_frames)
        self.future_frames = int(self.cfg.future_frames)
        self.min_history_frames = int(self.cfg.min_history_frames)
        self.dt = float(self.cfg.dt)
        self.include_ego = bool(self.cfg.include_ego)
        self.interpolate_single_frame_gaps = bool(self.cfg.interpolate_single_frame_gaps)
        self.max_agents = self.cfg.max_agents

        self.scene_to_tokens, self.sample_to_index, self.sample_to_scene = build_scene_timelines(self.nusc)
        self.sample_tokens = list(sample_tokens) if sample_tokens is not None else [
            token for scene_tokens in self.scene_to_tokens.values() for token in scene_tokens
        ]

        self._sample_cache: dict[str, dict[str, Any]] = {}
        self._annotation_cache: dict[str, dict[str, Any]] = {}
        self._annotation_state_cache: dict[str, AgentState] = {}
        self._ego_pose_cache: dict[str, tuple[np.ndarray, float]] = {}

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_token = self.sample_tokens[idx]
        sample = self._get_sample(sample_token)
        current_index = self.sample_to_index[sample_token]
        scene_token = self.sample_to_scene[sample_token]
        current_ego_xy, current_ego_yaw = self._get_ego_pose(sample_token)

        agents: list[dict[str, Any]] = []
        if self.include_ego:
            agents.append(
                self._build_agent_entry(
                    instance_token="ego",
                    current_token=sample_token,
                    class_name=EGO_CLASS_NAME,
                    history_states=self._build_ego_history(sample_token, current_index),
                    future_states=self._build_ego_future(sample_token, current_index),
                    current_ego_xy=current_ego_xy,
                    current_ego_yaw=current_ego_yaw,
                )
            )

        for ann_token in sample["anns"]:
            ann = self._get_annotation(ann_token)
            class_name = normalize_motion_agent_class(ann.get("category_name"))
            if class_name is None or class_name not in self.class_to_idx:
                continue
            agents.append(
                self._build_agent_entry(
                    instance_token=str(ann["instance_token"]),
                    current_token=ann_token,
                    class_name=class_name,
                    history_states=self._build_annotation_history(ann, current_index),
                    future_states=self._build_annotation_future(ann, current_index),
                    current_ego_xy=current_ego_xy,
                    current_ego_yaw=current_ego_yaw,
                )
            )

        if self.max_agents is not None and len(agents) > int(self.max_agents):
            ego_agents = [entry for entry in agents if entry["agent_token"] == "ego"]
            other_agents = [entry for entry in agents if entry["agent_token"] != "ego"]
            other_agents.sort(
                key=lambda entry: (
                    float(np.linalg.norm(entry["p_last_hist_ego"])),
                    str(entry["agent_token"]),
                )
            )
            agents = (ego_agents + other_agents)[: int(self.max_agents)]
        return self._stack_agents(sample_token=sample_token, scene_token=scene_token, agents=agents)

    def _get_sample(self, sample_token: str) -> dict[str, Any]:
        if sample_token not in self._sample_cache:
            self._sample_cache[sample_token] = self.nusc.get("sample", sample_token)
        return self._sample_cache[sample_token]

    def _get_annotation(self, ann_token: str) -> dict[str, Any]:
        if ann_token not in self._annotation_cache:
            self._annotation_cache[ann_token] = self.nusc.get("sample_annotation", ann_token)
        return self._annotation_cache[ann_token]

    def _get_ego_pose(self, sample_token: str) -> tuple[np.ndarray, float]:
        if sample_token not in self._ego_pose_cache:
            sample = self._get_sample(sample_token)
            lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            xy = np.asarray(ego_pose["translation"][:2], dtype=np.float32)
            yaw = float(quaternion_yaw(ego_pose["rotation"]))
            self._ego_pose_cache[sample_token] = (xy, yaw)
        return self._ego_pose_cache[sample_token]

    def _annotation_state(self, ann: dict[str, Any]) -> AgentState:
        ann_token = str(ann["token"])
        cached = self._annotation_state_cache.get(ann_token)
        if cached is not None:
            return cached

        state = AgentState(
            sample_token=str(ann["sample_token"]),
            scene_index=self.sample_to_index[str(ann["sample_token"])],
            instance_token=str(ann["instance_token"]),
            class_name=str(normalize_motion_agent_class(ann.get("category_name"))),
            xy_global=np.asarray(ann["translation"][:2], dtype=np.float32),
            yaw_global=float(quaternion_yaw(ann["rotation"])),
            size_xy=np.asarray(ann["size"][:2], dtype=np.float32),
            missing=False,
        )
        self._annotation_state_cache[ann_token] = state
        return state

    def _ego_state(self, sample_token: str) -> AgentState:
        xy_global, yaw_global = self._get_ego_pose(sample_token)
        return AgentState(
            sample_token=sample_token,
            scene_index=self.sample_to_index[sample_token],
            instance_token="ego",
            class_name=EGO_CLASS_NAME,
            xy_global=xy_global,
            yaw_global=yaw_global,
            size_xy=np.asarray(self.cfg.ego_size, dtype=np.float32),
            missing=False,
        )

    def _interpolate_state(self, first: AgentState, second: AgentState, alpha: float, scene_index: int) -> AgentState:
        scene_token = self.sample_to_scene[second.sample_token]
        return AgentState(
            sample_token=self.scene_to_tokens[scene_token][scene_index],
            scene_index=scene_index,
            instance_token=first.instance_token,
            class_name=first.class_name,
            xy_global=interpolate_xy(first.xy_global, second.xy_global, alpha),
            yaw_global=interpolate_angle(first.yaw_global, second.yaw_global, alpha),
            size_xy=interpolate_xy(first.size_xy, second.size_xy, alpha),
            missing=True,
        )

    def _build_annotation_history(self, ann: dict[str, Any], current_index: int) -> list[AgentState | None]:
        history: list[AgentState | None] = [None] * self.history_frames
        current = self._annotation_state(ann)
        history[-1] = current

        earliest = current_index - (self.history_frames - 1)
        last_state = current
        last_index = current_index
        prev_token = ann["prev"]

        while prev_token:
            prev_ann = self._get_annotation(prev_token)
            prev_state = self._annotation_state(prev_ann)
            gap = last_index - prev_state.scene_index
            if gap <= 0:
                break
            if gap > 2 or (gap == 2 and not self.interpolate_single_frame_gaps):
                break

            if gap == 2:
                mid_index = prev_state.scene_index + 1
                if mid_index >= earliest:
                    history[mid_index - earliest] = self._interpolate_state(prev_state, last_state, 0.5, mid_index)
            if prev_state.scene_index >= earliest:
                history[prev_state.scene_index - earliest] = prev_state

            last_state = prev_state
            last_index = prev_state.scene_index
            prev_token = prev_ann["prev"]
            if last_index <= earliest:
                break

        return history

    def _build_annotation_future(self, ann: dict[str, Any], current_index: int) -> list[AgentState | None]:
        future: list[AgentState | None] = [None] * self.future_frames
        last_state = self._annotation_state(ann)
        last_index = current_index
        next_token = ann["next"]
        max_index = current_index + self.future_frames

        while next_token:
            next_ann = self._get_annotation(next_token)
            next_state = self._annotation_state(next_ann)
            gap = next_state.scene_index - last_index
            if gap <= 0:
                break
            if gap > 2 or (gap == 2 and not self.interpolate_single_frame_gaps):
                break

            if gap == 2:
                mid_index = last_index + 1
                if mid_index <= max_index:
                    future[mid_index - current_index - 1] = self._interpolate_state(last_state, next_state, 0.5, mid_index)
            if next_state.scene_index <= max_index:
                future[next_state.scene_index - current_index - 1] = next_state

            last_state = next_state
            last_index = next_state.scene_index
            next_token = next_ann["next"]
            if last_index >= max_index:
                break

        return future

    def _build_ego_history(self, sample_token: str, current_index: int) -> list[AgentState | None]:
        history: list[AgentState | None] = [None] * self.history_frames
        scene_token = self.sample_to_scene[sample_token]
        earliest = current_index - (self.history_frames - 1)
        for scene_index in range(max(earliest, 0), current_index + 1):
            history[scene_index - earliest] = self._ego_state(self.scene_to_tokens[scene_token][scene_index])
        return history

    def _build_ego_future(self, sample_token: str, current_index: int) -> list[AgentState | None]:
        future: list[AgentState | None] = [None] * self.future_frames
        scene_token = self.sample_to_scene[sample_token]
        scene_tokens = self.scene_to_tokens[scene_token]
        for scene_index in range(current_index + 1, min(current_index + self.future_frames, len(scene_tokens) - 1) + 1):
            future[scene_index - current_index - 1] = self._ego_state(scene_tokens[scene_index])
        return future

    def _states_to_arrays(
        self,
        states: list[AgentState | None],
        current_ego_xy: np.ndarray,
        current_ego_yaw: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        seq_len = len(states)
        xy_ego = np.zeros((seq_len, 2), dtype=np.float32)
        yaw_ego = np.zeros((seq_len,), dtype=np.float32)
        size_xy = np.zeros((seq_len, 2), dtype=np.float32)
        valid = np.zeros((seq_len,), dtype=np.bool_)
        missing = np.zeros((seq_len,), dtype=np.bool_)

        for idx, state in enumerate(states):
            if state is None:
                continue
            xy_ego[idx] = global_xy_to_ego(state.xy_global, current_ego_xy, current_ego_yaw)
            yaw_ego[idx] = float(wrap_angle(state.yaw_global - current_ego_yaw))
            size_xy[idx] = state.size_xy
            valid[idx] = True
            missing[idx] = bool(state.missing)

        return xy_ego, yaw_ego, size_xy, valid, missing

    def _compute_velocity(self, xy_ego: np.ndarray, yaw_ego: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros((xy_ego.shape[0], 2), dtype=np.float32)
        yaw_rate = np.zeros((xy_ego.shape[0],), dtype=np.float32)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size <= 1:
            return velocity, yaw_rate

        for pos, idx in enumerate(valid_idx):
            prev_idx = int(valid_idx[max(pos - 1, 0)])
            next_idx = int(valid_idx[min(pos + 1, valid_idx.size - 1)])
            if prev_idx == next_idx:
                continue
            delta_t = max((next_idx - prev_idx) * self.dt, 1e-6)
            velocity[idx] = (xy_ego[next_idx] - xy_ego[prev_idx]) / delta_t
            yaw_rate[idx] = float(wrap_angle(yaw_ego[next_idx] - yaw_ego[prev_idx]) / delta_t)

        first = int(valid_idx[0])
        last = int(valid_idx[-1])
        velocity[:first] = velocity[first]
        velocity[last + 1 :] = velocity[last]
        yaw_rate[:first] = yaw_rate[first]
        yaw_rate[last + 1 :] = yaw_rate[last]
        return velocity, yaw_rate

    def _reference_history_index(self, valid: np.ndarray, missing: np.ndarray) -> int:
        observed_idx = np.flatnonzero(valid & ~missing)
        if observed_idx.size > 0:
            return int(observed_idx[-1])

        valid_idx = np.flatnonzero(valid)
        if valid_idx.size > 0:
            return int(valid_idx[-1])

        return int(len(valid) - 1)

    def _build_agent_entry(
        self,
        instance_token: str,
        current_token: str,
        class_name: str,
        history_states: list[AgentState | None],
        future_states: list[AgentState | None],
        current_ego_xy: np.ndarray,
        current_ego_yaw: float,
    ) -> dict[str, Any]:
        history_xy, history_yaw, history_size, history_valid, history_missing = self._states_to_arrays(
            history_states,
            current_ego_xy=current_ego_xy,
            current_ego_yaw=current_ego_yaw,
        )
        future_xy, future_yaw, _, future_valid, future_missing = self._states_to_arrays(
            future_states,
            current_ego_xy=current_ego_xy,
            current_ego_yaw=current_ego_yaw,
        )
        history_vel, history_yaw_rate = self._compute_velocity(history_xy, history_yaw, history_valid)

        class_onehot = np.zeros((len(self.class_names),), dtype=np.float32)
        class_onehot[self.class_to_idx[class_name]] = 1.0
        history_features = []
        for step in range(self.history_frames):
            history_features.append(
                np.concatenate(
                    [
                        history_xy[step],
                        np.asarray([np.cos(history_yaw[step]), np.sin(history_yaw[step])], dtype=np.float32),
                        history_vel[step],
                        np.asarray([history_yaw_rate[step]], dtype=np.float32),
                        history_size[step],
                        class_onehot,
                        np.asarray([float(history_valid[step]), float(history_missing[step])], dtype=np.float32),
                    ],
                    axis=0,
                )
            )

        history_observed = history_valid & ~history_missing
        future_observed = future_valid & ~future_missing
        hist_len = int(history_observed.sum())
        future_gt_len = int(future_observed.sum())
        future_len = int(future_valid.sum())

        ref_idx = self._reference_history_index(history_valid, history_missing)
        current_xy = history_xy[ref_idx]
        current_yaw = float(history_yaw[ref_idx])
        current_size = history_size[ref_idx]
        current_speed = float(np.linalg.norm(history_vel[ref_idx]))
        rotation_local_to_ego = yaw_to_rotation_matrix(current_yaw)

        return {
            "agent_token": instance_token,
            "current_token": current_token,
            "class_id": self.class_to_idx[class_name],
            "history_features": np.stack(history_features, axis=0).astype(np.float32, copy=False),
            "history_positions_ego": history_xy.astype(np.float32, copy=False),
            "history_yaw_ego": history_yaw.astype(np.float32, copy=False),
            "history_vel_xy": history_vel.astype(np.float32, copy=False),
            "history_yaw_rate": history_yaw_rate.astype(np.float32, copy=False),
            "history_valid_mask": history_valid,
            "history_missing_mask": history_missing,
            "history_len": hist_len,
            "future_positions_ego": future_xy.astype(np.float32, copy=False),
            "future_yaw_ego": future_yaw.astype(np.float32, copy=False),
            "future_valid_mask": future_valid,
            "future_missing_mask": future_missing,
            "future_len": future_len,
            "future_gt_len": future_gt_len,
            "current_xy_ego": current_xy.astype(np.float32, copy=False),
            "p_last_hist_ego": current_xy.astype(np.float32, copy=False),
            "current_yaw_ego": np.float32(current_yaw),
            "agent_yaw_ego": np.float32(current_yaw),
            "current_size_xy": current_size.astype(np.float32, copy=False),
            "current_speed": np.float32(current_speed),
            "rotation_local_to_ego": rotation_local_to_ego.astype(np.float32, copy=False),
            "rotation_ego_to_local": rotation_local_to_ego.T.astype(np.float32, copy=False),
            "train_valid": bool(hist_len >= self.min_history_frames and future_gt_len == self.future_frames),
            "inference_valid": bool(hist_len >= self.min_history_frames),
        }

    def _stack_agents(self, sample_token: str, scene_token: str, agents: list[dict[str, Any]]) -> dict[str, Any]:
        feature_dim = 2 + 2 + 2 + 1 + 2 + len(self.class_names) + 2
        if not agents:
            return {
                "sample_token": sample_token,
                "scene_token": scene_token,
                "num_agents": 0,
                "agent_tokens": [],
                "class_ids": torch.zeros((0,), dtype=torch.long),
                "history_features": torch.zeros((0, self.history_frames, feature_dim), dtype=torch.float32),
                "history_positions_ego": torch.zeros((0, self.history_frames, 2), dtype=torch.float32),
                "history_yaw_ego": torch.zeros((0, self.history_frames), dtype=torch.float32),
                "history_vel_xy": torch.zeros((0, self.history_frames, 2), dtype=torch.float32),
                "history_yaw_rate": torch.zeros((0, self.history_frames), dtype=torch.float32),
                "history_valid_mask": torch.zeros((0, self.history_frames), dtype=torch.bool),
                "history_missing_mask": torch.zeros((0, self.history_frames), dtype=torch.bool),
                "history_len": torch.zeros((0,), dtype=torch.long),
                "future_positions_ego": torch.zeros((0, self.future_frames, 2), dtype=torch.float32),
                "future_yaw_ego": torch.zeros((0, self.future_frames), dtype=torch.float32),
                "future_valid_mask": torch.zeros((0, self.future_frames), dtype=torch.bool),
                "future_missing_mask": torch.zeros((0, self.future_frames), dtype=torch.bool),
                "future_len": torch.zeros((0,), dtype=torch.long),
                "future_gt_len": torch.zeros((0,), dtype=torch.long),
                "current_xy_ego": torch.zeros((0, 2), dtype=torch.float32),
                "p_last_hist_ego": torch.zeros((0, 2), dtype=torch.float32),
                "current_yaw_ego": torch.zeros((0,), dtype=torch.float32),
                "agent_yaw_ego": torch.zeros((0,), dtype=torch.float32),
                "current_size_xy": torch.zeros((0, 2), dtype=torch.float32),
                "current_speed": torch.zeros((0,), dtype=torch.float32),
                "rotation_local_to_ego": torch.zeros((0, 2, 2), dtype=torch.float32),
                "rotation_ego_to_local": torch.zeros((0, 2, 2), dtype=torch.float32),
                "train_valid_mask": torch.zeros((0,), dtype=torch.bool),
                "inference_valid_mask": torch.zeros((0,), dtype=torch.bool),
            }

        def stack(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(np.stack([agent[name] for agent in agents], axis=0), dtype=dtype)

        return {
            "sample_token": sample_token,
            "scene_token": scene_token,
            "num_agents": len(agents),
            "agent_tokens": [agent["agent_token"] for agent in agents],
            "class_ids": torch.as_tensor([agent["class_id"] for agent in agents], dtype=torch.long),
            "history_features": stack("history_features", torch.float32),
            "history_positions_ego": stack("history_positions_ego", torch.float32),
            "history_yaw_ego": stack("history_yaw_ego", torch.float32),
            "history_vel_xy": stack("history_vel_xy", torch.float32),
            "history_yaw_rate": stack("history_yaw_rate", torch.float32),
            "history_valid_mask": stack("history_valid_mask", torch.bool),
            "history_missing_mask": stack("history_missing_mask", torch.bool),
            "history_len": torch.as_tensor([agent["history_len"] for agent in agents], dtype=torch.long),
            "future_positions_ego": stack("future_positions_ego", torch.float32),
            "future_yaw_ego": stack("future_yaw_ego", torch.float32),
            "future_valid_mask": stack("future_valid_mask", torch.bool),
            "future_missing_mask": stack("future_missing_mask", torch.bool),
            "future_len": torch.as_tensor([agent["future_len"] for agent in agents], dtype=torch.long),
            "future_gt_len": torch.as_tensor([agent["future_gt_len"] for agent in agents], dtype=torch.long),
            "current_xy_ego": stack("current_xy_ego", torch.float32),
            "p_last_hist_ego": stack("p_last_hist_ego", torch.float32),
            "current_yaw_ego": torch.as_tensor([agent["current_yaw_ego"] for agent in agents], dtype=torch.float32),
            "agent_yaw_ego": torch.as_tensor([agent["agent_yaw_ego"] for agent in agents], dtype=torch.float32),
            "current_size_xy": stack("current_size_xy", torch.float32),
            "current_speed": torch.as_tensor([agent["current_speed"] for agent in agents], dtype=torch.float32),
            "rotation_local_to_ego": stack("rotation_local_to_ego", torch.float32),
            "rotation_ego_to_local": stack("rotation_ego_to_local", torch.float32),
            "train_valid_mask": torch.as_tensor([agent["train_valid"] for agent in agents], dtype=torch.bool),
            "inference_valid_mask": torch.as_tensor([agent["inference_valid"] for agent in agents], dtype=torch.bool),
        }


def collate_motion_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_agents = max((item["num_agents"] for item in batch), default=0)
    history_frames = batch[0]["history_features"].shape[1] if batch else 0
    history_dim = batch[0]["history_features"].shape[2] if batch and batch[0]["history_features"].ndim == 3 else 0
    future_frames = batch[0]["future_positions_ego"].shape[1] if batch else 0

    def zeros(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    packed = {
        "class_ids": torch.full((batch_size, max_agents), -1, dtype=torch.long),
        "history_features": zeros(batch_size, max_agents, history_frames, history_dim, dtype=torch.float32),
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
    }

    sample_tokens: list[str] = []
    scene_tokens: list[str] = []
    agent_tokens: list[list[str]] = []

    for batch_idx, item in enumerate(batch):
        num_agents = int(item["num_agents"])
        sample_tokens.append(item["sample_token"])
        scene_tokens.append(item["scene_token"])
        agent_tokens.append(list(item["agent_tokens"]))
        if num_agents == 0:
            continue

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

    packed["sample_tokens"] = sample_tokens
    packed["scene_tokens"] = scene_tokens
    packed["agent_tokens"] = agent_tokens
    return packed


def build_loader(
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
        "collate_fn": collate_motion_batch,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)
