from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import heapq
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.nuscenes_utils import build_scene_timelines
from motion_v1.categories import AGENT_CLASS_NAMES, EGO_CLASS_NAME, normalize_motion_agent_class
from motion_v1.geometry import global_xy_to_ego, quaternion_yaw, wrap_angle


CURRENT_ARTIFACT_SEMANTIC_VERSION = 2


@dataclass(frozen=True)
class V1DataConfig:
    history_frames: int = 4
    future_frames: int = 12
    stride: int = 2
    dt: float = 0.5
    max_agents: int | None = None
    max_polylines: int = 128
    max_objects: int = 32
    max_polyline_points: int = 20
    map_radius_m: float = 80.0
    history_feature_dim: int = 18
    polyline_point_dim: int = 4
    polyline_attr_dim: int = 8
    object_feature_dim: int = 10
    anchor_k12: int = 64
    anchor_kmeans_iters: int = 25
    anchor_random_seed: int = 13
    anchor_stationary_threshold_m: float = 0.5


@dataclass(frozen=True)
class V1AugmentationConfig:
    enabled: bool = False
    apply_prob: float = 0.8
    max_rotation_deg: float = 22.5
    translation_std_m: float = 1.0


@dataclass(frozen=True)
class V1ArtifactMetadata:
    split_name: str
    history_frames: int
    future_frames: int
    stride: int
    history_feature_dim: int
    polyline_point_dim: int
    polyline_attr_dim: int
    object_feature_dim: int
    artifact_semantic_version: int = CURRENT_ARTIFACT_SEMANTIC_VERSION


@dataclass(frozen=True)
class _Window:
    scene_token: str
    start_idx: int


@dataclass(frozen=True)
class _PolylineRecord:
    token: str
    points_global: np.ndarray
    attrs: np.ndarray
    centroid_global: np.ndarray


@dataclass(frozen=True)
class _ObjectRecord:
    token: str
    xy_global: np.ndarray
    yaw_global: float
    size_xy: np.ndarray
    type_onehot: np.ndarray


def select_split_scene_tokens(
    nusc,
    split_name: str,
    scene_limit: int | None = None,
) -> list[str]:
    from nuscenes.utils.splits import create_splits_scenes

    split_scene_names = set(create_splits_scenes()[split_name])
    scenes = [scene for scene in nusc.scene if scene["name"] in split_scene_names]
    if scene_limit is not None:
        scenes = scenes[: int(scene_limit)]
    return [str(scene["token"]) for scene in scenes]


def _random_uniform(low: float, high: float) -> float:
    return float(torch.empty((), dtype=torch.float32).uniform_(low, high).item())


def _random_normal(std: float) -> float:
    if std <= 0.0:
        return 0.0
    return float(torch.randn((), dtype=torch.float32).item() * std)


def _rotation_matrix(angle_rad: float) -> torch.Tensor:
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    return torch.tensor(
        [
            [cos_angle, sin_angle],
            [-sin_angle, cos_angle],
        ],
        dtype=torch.float32,
    )


def _apply_scene_augmentation(sample: dict[str, Any], cfg: V1AugmentationConfig | None) -> dict[str, Any]:
    if cfg is None or not cfg.enabled:
        return sample

    if cfg.apply_prob <= 0.0:
        return sample
    if cfg.apply_prob < 1.0 and float(torch.rand((), dtype=torch.float32).item()) > float(cfg.apply_prob):
        return sample

    angle_rad = math.radians(_random_uniform(-float(cfg.max_rotation_deg), float(cfg.max_rotation_deg)))
    rotation = _rotation_matrix(angle_rad)
    translation = torch.tensor(
        [_random_normal(float(cfg.translation_std_m)), _random_normal(float(cfg.translation_std_m))],
        dtype=torch.float32,
    )

    augmented = dict(sample)

    history_features = sample["history_features"].clone()
    history_features[..., 0:2] = history_features[..., 0:2] @ rotation + translation.view(1, 1, 2)
    history_features[..., 2:4] = history_features[..., 2:4] @ rotation
    history_features[..., 4:6] = history_features[..., 4:6] @ rotation
    augmented["history_features"] = history_features

    future_positions_ego = sample["future_positions_ego"].clone()
    future_positions_ego = future_positions_ego @ rotation + translation.view(1, 1, 2)
    augmented["future_positions_ego"] = future_positions_ego

    polyline_point_features = sample["polyline_point_features"].clone()
    polyline_point_features[..., 0:2] = polyline_point_features[..., 0:2] @ rotation + translation.view(1, 1, 2)
    polyline_point_features[..., 2:4] = polyline_point_features[..., 2:4] @ rotation
    augmented["polyline_point_features"] = polyline_point_features

    object_features = sample["object_features"].clone()
    if object_features.numel() > 0:
        object_features[..., 0:2] = object_features[..., 0:2] @ rotation + translation.view(1, 2)
        object_features[..., 2:4] = object_features[..., 2:4] @ rotation
    augmented["object_features"] = object_features

    return augmented


def _geometry_pose_and_size(geometry) -> tuple[np.ndarray, float, np.ndarray]:
    from shapely.geometry import LineString, Polygon

    if isinstance(geometry, Polygon):
        centroid = np.asarray(geometry.centroid.coords[0][:2], dtype=np.float32)
        rect = geometry.minimum_rotated_rectangle
        rect_points = np.asarray(rect.exterior.coords[:4], dtype=np.float32)
        edges = rect_points[(np.arange(4) + 1) % 4] - rect_points[np.arange(4)]
        lengths = np.linalg.norm(edges, axis=-1)
        major_idx = int(lengths.argmax())
        major_edge = edges[major_idx]
        yaw = float(np.arctan2(major_edge[1], major_edge[0]))
        size_xy = np.asarray([lengths.min(), lengths.max()], dtype=np.float32)
        return centroid, yaw, size_xy

    if isinstance(geometry, LineString):
        coords = np.asarray(geometry.coords, dtype=np.float32)[:, :2].astype(np.float32, copy=False)
        centroid = coords.mean(axis=0)
        delta = coords[-1] - coords[0]
        yaw = float(np.arctan2(delta[1], delta[0]))
        size_xy = np.asarray([0.5, max(float(np.linalg.norm(delta)), 0.5)], dtype=np.float32)
        return centroid, yaw, size_xy

    raise TypeError(f"Unsupported geometry type: {type(geometry)!r}")


def _polyline_curvature(points_global: np.ndarray) -> float:
    segment = np.diff(points_global, axis=0)
    seg_len = np.linalg.norm(segment, axis=-1)
    total_len = float(seg_len.sum())
    if total_len < 1e-6:
        return 0.0
    headings = np.arctan2(segment[:, 1], segment[:, 0])
    heading_delta = wrap_angle(headings[1:] - headings[:-1]) if headings.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
    return float(np.abs(heading_delta).sum() / total_len)


def _build_polyline_attrs(
    polyline_type: str,
    points_global: np.ndarray,
    width: float,
    connected_count: int,
    is_fork: bool,
) -> np.ndarray:
    type_onehot = {
        "lane": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "lane_connector": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "divider": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    }[polyline_type]
    length = float(np.linalg.norm(np.diff(points_global, axis=0), axis=-1).sum())
    curvature = _polyline_curvature(points_global)
    return np.concatenate(
        [
            type_onehot,
            np.asarray(
                [
                    float(width),
                    float(length),
                    float(curvature),
                    float(connected_count),
                    float(is_fork),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _resample_polyline(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    segment = np.diff(points, axis=0)
    seg_len = np.linalg.norm(segment, axis=-1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(arc[-1])
    if total < 1e-6:
        raise ValueError("Polyline arc length must be positive.")
    target = np.linspace(0.0, total, num=max_points, dtype=np.float32)
    x = np.interp(target, arc, points[:, 0]).astype(np.float32)
    y = np.interp(target, arc, points[:, 1]).astype(np.float32)
    return np.stack([x, y], axis=-1)


def _polyline_points_to_features(
    points_global: np.ndarray,
    ego_xy_global: np.ndarray,
    ego_yaw_global: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_points = np.stack(
        [global_xy_to_ego(point, ego_xy_global, ego_yaw_global) for point in points_global],
        axis=0,
    ).astype(np.float32, copy=False)
    resampled = _resample_polyline(local_points, max_points)
    num_points = resampled.shape[0]
    tangent = np.zeros_like(resampled)
    tangent[1:-1] = 0.5 * (resampled[2:] - resampled[:-2])
    tangent[0] = resampled[1] - resampled[0]
    tangent[-1] = resampled[-1] - resampled[-2]

    point_features = np.zeros((num_points, 4), dtype=np.float32)
    point_features[:, 0:2] = resampled
    point_features[:, 2:4] = tangent
    point_mask = np.ones((num_points,), dtype=np.bool_)
    return point_features, point_mask


def _object_feature(record: _ObjectRecord, ego_xy_global: np.ndarray, ego_yaw_global: float) -> np.ndarray:
    xy_ego = global_xy_to_ego(record.xy_global, ego_xy_global, ego_yaw_global)
    yaw_ego = float(wrap_angle(record.yaw_global - ego_yaw_global))
    return np.concatenate(
        [
            xy_ego.astype(np.float32, copy=False),
            np.asarray([math.cos(yaw_ego), math.sin(yaw_ego)], dtype=np.float32),
            record.size_xy.astype(np.float32, copy=False),
            record.type_onehot.astype(np.float32, copy=False),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def build_v1_map_store(
    nusc,
    cfg: V1DataConfig | None = None,
    map_names: list[str] | None = None,
) -> dict[str, dict[str, list[Any]]]:
    cfg = cfg or V1DataConfig()
    try:
        from nuscenes.map_expansion.map_api import NuScenesMap
    except ImportError as exc:
        raise ImportError("Building V1 map store requires nuscenes map devkit.") from exc

    if map_names is None:
        names = {str(nusc.get("log", scene["log_token"])["location"]) for scene in nusc.scene}
        map_names = sorted(names)

    store_by_name: dict[str, dict[str, list[Any]]] = {}
    for map_name in map_names:
        nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)
        polylines: list[_PolylineRecord] = []
        objects: list[_ObjectRecord] = []

        for layer_name, poly_type in (("lane", "lane"), ("lane_connector", "lane_connector")):
            layer_records = getattr(nusc_map, layer_name, [])
            tokens = [str(record["token"]) for record in layer_records]
            discretized = nusc_map.discretize_lanes(tokens, 1.0) if tokens else {}
            for record in layer_records:
                token = str(record["token"])
                points_global = np.asarray(discretized[token], dtype=np.float32)[:, :2].astype(np.float32, copy=False)
                outgoing = list(nusc_map.get_outgoing_lane_ids(token))
                incoming = list(nusc_map.get_incoming_lane_ids(token))
                connected_count = len(outgoing) + len(incoming)
                attrs = _build_polyline_attrs(
                    polyline_type=poly_type,
                    points_global=points_global,
                    width=float(record.get("width", 4.0)),
                    connected_count=connected_count,
                    is_fork=connected_count > 2,
                )
                polylines.append(
                    _PolylineRecord(
                        token=token,
                        points_global=points_global,
                        attrs=attrs,
                        centroid_global=points_global.mean(axis=0).astype(np.float32, copy=False),
                    )
                )

        for layer_name in ("road_divider", "lane_divider"):
            for record in getattr(nusc_map, layer_name, []):
                line = nusc_map.extract_line(str(record["line_token"]))
                points_global = np.asarray(line.coords, dtype=np.float32)[:, :2].astype(np.float32, copy=False)
                token = str(record["token"])
                attrs = _build_polyline_attrs(
                    polyline_type="divider",
                    points_global=points_global,
                    width=0.5,
                    connected_count=0,
                    is_fork=False,
                )
                polylines.append(
                    _PolylineRecord(
                        token=token,
                        points_global=points_global,
                        attrs=attrs,
                        centroid_global=points_global.mean(axis=0).astype(np.float32, copy=False),
                    )
                )

        object_layers = {
            "ped_crossing": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "stop_line": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "traffic_light": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "carpark_area": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
        for layer_name, type_onehot in object_layers.items():
            for record in getattr(nusc_map, layer_name, []):
                polygon_token = record.get("polygon_token")
                polygon_tokens = record.get("polygon_tokens")
                if polygon_token:
                    geometry = nusc_map.extract_polygon(str(polygon_token))
                elif polygon_tokens:
                    geometry = nusc_map.extract_polygon(str(polygon_tokens[0]))
                elif record.get("line_token"):
                    geometry = nusc_map.extract_line(str(record["line_token"]))
                else:
                    raise KeyError("Map record does not contain polygon or line token.")
                xy_global, yaw_global, size_xy = _geometry_pose_and_size(geometry)
                objects.append(
                    _ObjectRecord(
                        token=str(record["token"]),
                        xy_global=xy_global.astype(np.float32, copy=False),
                        yaw_global=float(yaw_global),
                        size_xy=size_xy.astype(np.float32, copy=False),
                        type_onehot=type_onehot.copy(),
                    )
                )

        store_by_name[map_name] = {"polylines": polylines, "objects": objects}
    return store_by_name


class V1WindowDataset(Dataset):
    def __init__(
        self,
        nusc,
        map_store_by_name: dict[str, dict[str, list[Any]]],
        cfg: V1DataConfig | None = None,
        scene_tokens: list[str] | None = None,
        augmentation: V1AugmentationConfig | None = None,
    ) -> None:
        self.nusc = nusc
        self.cfg = cfg or V1DataConfig()
        self.augmentation = augmentation
        self.map_store_by_name = map_store_by_name
        self.class_names = tuple(AGENT_CLASS_NAMES)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        scene_to_tokens, _, _ = build_scene_timelines(self.nusc)
        self.scene_tokens = list(scene_tokens) if scene_tokens is not None else list(scene_to_tokens.keys())
        self.scene_to_tokens = {scene_token: scene_to_tokens[scene_token] for scene_token in self.scene_tokens}

        self._scene_map_name: dict[str, str] = {}
        self._heap_counter = 0

        self.scene_agent_frames: dict[str, list[dict[str, dict[str, Any]]]] = {}
        self.scene_ego_frames: dict[str, list[dict[str, Any]]] = {}
        self._index_scenes()
        self.windows = self._build_windows()

    def _index_scenes(self) -> None:
        for scene_token, sample_tokens in self.scene_to_tokens.items():
            scene = self.nusc.get("scene", scene_token)
            log = self.nusc.get("log", scene["log_token"])
            self._scene_map_name[scene_token] = str(log["location"])

            agent_frames: list[dict[str, dict[str, Any]]] = []
            ego_frames: list[dict[str, Any]] = []

            for sample_token in sample_tokens:
                sample = self.nusc.get("sample", sample_token)
                lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
                ego_pose = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
                ego_xy = np.asarray(ego_pose["translation"][:2], dtype=np.float32)
                ego_yaw = float(quaternion_yaw(ego_pose["rotation"]))
                ego_frames.append(
                    {
                        "sample_token": sample_token,
                        "xy_global": ego_xy,
                        "yaw_global": ego_yaw,
                        "size_xy": np.asarray([1.9, 4.8], dtype=np.float32),
                        "class_name": EGO_CLASS_NAME,
                    }
                )

                frame_agents: dict[str, dict[str, Any]] = {}
                for ann_token in sample["anns"]:
                    ann = self.nusc.get("sample_annotation", ann_token)
                    class_name = normalize_motion_agent_class(ann.get("category_name"))
                    if class_name is None:
                        continue
                    frame_agents[str(ann["instance_token"])] = {
                        "sample_token": sample_token,
                        "xy_global": np.asarray(ann["translation"][:2], dtype=np.float32),
                        "yaw_global": float(quaternion_yaw(ann["rotation"])),
                        "size_xy": np.asarray(ann["size"][:2], dtype=np.float32),
                        "class_name": class_name,
                    }
                agent_frames.append(frame_agents)

            self.scene_agent_frames[scene_token] = agent_frames
            self.scene_ego_frames[scene_token] = ego_frames

    def _build_windows(self) -> list[_Window]:
        windows: list[_Window] = []
        total = self.cfg.history_frames + self.cfg.future_frames
        for scene_token, sample_tokens in self.scene_to_tokens.items():
            last_start = len(sample_tokens) - total
            for start_idx in range(0, max(last_start + 1, 0), self.cfg.stride):
                windows.append(_Window(scene_token=scene_token, start_idx=start_idx))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def _build_agent_entry(
        self,
        agent_token: str,
        class_name: str,
        history_states: list[dict[str, Any]],
        future_states: list[dict[str, Any]],
        ego_xy_global: np.ndarray,
        ego_yaw_global: float,
    ) -> dict[str, Any]:
        history_xy = np.stack(
            [global_xy_to_ego(state["xy_global"], ego_xy_global, ego_yaw_global) for state in history_states],
            axis=0,
        ).astype(np.float32, copy=False)
        future_xy = np.stack(
            [global_xy_to_ego(state["xy_global"], ego_xy_global, ego_yaw_global) for state in future_states],
            axis=0,
        ).astype(np.float32, copy=False)
        history_yaw = np.asarray(
            [wrap_angle(float(state["yaw_global"]) - ego_yaw_global) for state in history_states],
            dtype=np.float32,
        )
        history_size = np.stack([state["size_xy"] for state in history_states], axis=0).astype(np.float32, copy=False)

        history_vel = np.zeros_like(history_xy, dtype=np.float32)
        history_yaw_rate = np.zeros((history_yaw.shape[0],), dtype=np.float32)
        history_vel[0] = (history_xy[1] - history_xy[0]) / self.cfg.dt
        history_vel[-1] = (history_xy[-1] - history_xy[-2]) / self.cfg.dt
        history_yaw_rate[0] = float(wrap_angle(history_yaw[1] - history_yaw[0]) / self.cfg.dt)
        history_yaw_rate[-1] = float(wrap_angle(history_yaw[-1] - history_yaw[-2]) / self.cfg.dt)
        for idx in range(1, history_xy.shape[0] - 1):
            history_vel[idx] = (history_xy[idx + 1] - history_xy[idx - 1]) / (2.0 * self.cfg.dt)
            history_yaw_rate[idx] = float(
                wrap_angle(history_yaw[idx + 1] - history_yaw[idx - 1]) / (2.0 * self.cfg.dt)
            )

        class_id = self.class_to_idx[class_name]
        class_onehot = np.zeros((len(self.class_names),), dtype=np.float32)
        class_onehot[class_id] = 1.0
        history_features = np.concatenate(
            [
                history_xy,
                np.stack([np.cos(history_yaw), np.sin(history_yaw)], axis=-1).astype(np.float32, copy=False),
                history_vel.astype(np.float32, copy=False),
                history_yaw_rate[:, None].astype(np.float32, copy=False),
                history_size,
                np.broadcast_to(class_onehot[None, :], (history_xy.shape[0], class_onehot.shape[0])),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

        return {
            "agent_token": agent_token,
            "class_id": class_id,
            "history_features": history_features,
            "future_positions_ego": future_xy.astype(np.float32, copy=False),
        }

    def _select_topk_records(self, records: list[Any], k: int, distance_fn) -> list[Any]:
        if k <= 0 or not records:
            return []
        heap: list[tuple[float, int, Any]] = []
        for record in records:
            dist = float(distance_fn(record))
            if dist > self.cfg.map_radius_m:
                continue
            item = (-dist, self._heap_counter, record)
            self._heap_counter += 1
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, item)
        return [record for _, _, record in sorted(heap, key=lambda item: (-item[0], item[1]))]

    def _collect_agents(
        self,
        scene_token: str,
        start_idx: int,
        hist_end_idx: int,
        ego_xy_global: np.ndarray,
        ego_yaw_global: float,
    ) -> list[dict[str, Any]]:
        hist_end = start_idx + self.cfg.history_frames
        fut_end = hist_end + self.cfg.future_frames

        agents: list[dict[str, Any]] = []
        ego_frames = self.scene_ego_frames[scene_token]
        agents.append(
            self._build_agent_entry(
                agent_token="ego",
                class_name=EGO_CLASS_NAME,
                history_states=ego_frames[start_idx:hist_end],
                future_states=ego_frames[hist_end:fut_end],
                ego_xy_global=ego_xy_global,
                ego_yaw_global=ego_yaw_global,
            )
        )

        for instance_token, current_state in self.scene_agent_frames[scene_token][hist_end_idx].items():
            track: list[dict[str, Any]] = []
            for frame_idx in range(start_idx, fut_end):
                state = self.scene_agent_frames[scene_token][frame_idx].get(instance_token)
                if state is None:
                    track = []
                    break
                track.append(state)
            if not track:
                continue
            agents.append(
                self._build_agent_entry(
                    agent_token=instance_token,
                    class_name=str(current_state["class_name"]),
                    history_states=track[: self.cfg.history_frames],
                    future_states=track[self.cfg.history_frames :],
                    ego_xy_global=ego_xy_global,
                    ego_yaw_global=ego_yaw_global,
                )
            )

        if self.cfg.max_agents is None or len(agents) <= self.cfg.max_agents:
            return agents

        ego_agents = agents[:1]
        other_agents = agents[1:]
        other_agents.sort(key=lambda agent: float(np.linalg.norm(agent["history_features"][-1, 0:2])))
        return (ego_agents + other_agents)[: self.cfg.max_agents]

    def _pack_agents(self, agents: list[dict[str, Any]]) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            [agent["agent_token"] for agent in agents],
            torch.as_tensor([agent["class_id"] for agent in agents], dtype=torch.long),
            torch.from_numpy(np.stack([agent["history_features"] for agent in agents], axis=0)),
            torch.from_numpy(np.stack([agent["future_positions_ego"] for agent in agents], axis=0)),
        )

    def _select_local_map(
        self,
        map_name: str,
        ego_xy_global: np.ndarray,
        ego_yaw_global: float,
    ) -> dict[str, torch.Tensor]:
        store = self.map_store_by_name[map_name]
        polylines = self._select_topk_records(
            store.get("polylines", []),
            self.cfg.max_polylines,
            lambda record: np.linalg.norm(record.centroid_global - ego_xy_global),
        )
        objects = self._select_topk_records(
            store.get("objects", []),
            self.cfg.max_objects,
            lambda record: np.linalg.norm(record.xy_global - ego_xy_global),
        )

        num_polylines = len(polylines)
        polyline_point_features = np.zeros(
            (num_polylines, self.cfg.max_polyline_points, self.cfg.polyline_point_dim),
            dtype=np.float32,
        )
        polyline_point_mask = np.zeros((num_polylines, self.cfg.max_polyline_points), dtype=np.bool_)
        polyline_attrs = np.zeros((num_polylines, self.cfg.polyline_attr_dim), dtype=np.float32)

        for poly_idx, record in enumerate(polylines):
            points, mask = _polyline_points_to_features(
                record.points_global,
                ego_xy_global=ego_xy_global,
                ego_yaw_global=ego_yaw_global,
                max_points=self.cfg.max_polyline_points,
            )
            num_points = min(points.shape[0], self.cfg.max_polyline_points)
            polyline_point_features[poly_idx, :num_points] = points[:num_points]
            polyline_point_mask[poly_idx, :num_points] = mask[:num_points]
            polyline_attrs[poly_idx] = record.attrs[: self.cfg.polyline_attr_dim]

        num_objects = len(objects)
        object_features = np.zeros((num_objects, self.cfg.object_feature_dim), dtype=np.float32)
        for obj_idx, record in enumerate(objects):
            object_features[obj_idx] = _object_feature(record, ego_xy_global, ego_yaw_global)[: self.cfg.object_feature_dim]

        return {
            "num_polylines": int(num_polylines),
            "num_objects": int(num_objects),
            "polyline_point_features": torch.from_numpy(polyline_point_features),
            "polyline_point_mask": torch.from_numpy(polyline_point_mask),
            "polyline_attrs": torch.from_numpy(polyline_attrs),
            "object_features": torch.from_numpy(object_features),
        }

    def _build_sample(self, idx: int) -> dict[str, Any]:
        window = self.windows[idx]
        scene_token = window.scene_token
        start_idx = window.start_idx
        hist_end_idx = start_idx + self.cfg.history_frames - 1
        ref_token = self.scene_to_tokens[scene_token][hist_end_idx]
        ego_state = self.scene_ego_frames[scene_token][hist_end_idx]
        ego_xy_global = ego_state["xy_global"]
        ego_yaw_global = float(ego_state["yaw_global"])
        map_name = self._scene_map_name[scene_token]

        agents = self._collect_agents(
            scene_token=scene_token,
            start_idx=start_idx,
            hist_end_idx=hist_end_idx,
            ego_xy_global=ego_xy_global,
            ego_yaw_global=ego_yaw_global,
        )
        agent_tokens, class_ids, history_features, future_positions_ego = self._pack_agents(agents)
        map_tensors = self._select_local_map(map_name, ego_xy_global, ego_yaw_global)
        return {
            "scene_token": scene_token,
            "sample_token": ref_token,
            "window_start_idx": start_idx,
            "map_name": map_name,
            "num_agents": int(len(agents)),
            "agent_tokens": agent_tokens,
            "class_ids": class_ids,
            "history_features": history_features,
            "future_positions_ego": future_positions_ego,
            **map_tensors,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return _apply_scene_augmentation(
            self._build_sample(idx),
            self.augmentation,
        )

    def build_artifact_payload(
        self,
        split_name: str,
        build_anchor_bank: bool = False,
    ) -> dict[str, Any]:
        samples = [self._build_sample(idx) for idx in range(len(self))]
        payload: dict[str, Any] = {
            "metadata": asdict(
                V1ArtifactMetadata(
                    split_name=split_name,
                    history_frames=self.cfg.history_frames,
                    future_frames=self.cfg.future_frames,
                    stride=self.cfg.stride,
                    history_feature_dim=self.cfg.history_feature_dim,
                    polyline_point_dim=self.cfg.polyline_point_dim,
                    polyline_attr_dim=self.cfg.polyline_attr_dim,
                    object_feature_dim=self.cfg.object_feature_dim,
                )
            ),
            "samples": samples,
        }
        if build_anchor_bank:
            payload["anchor_bank"] = build_anchor_bank_kmeans(samples, self.cfg)
        _validate_artifact_payload(payload)
        return payload

    def save_artifact(
        self,
        output_path: str | Path,
        split_name: str,
        build_anchor_bank: bool = False,
    ) -> dict[str, Any]:
        payload = self.build_artifact_payload(split_name=split_name, build_anchor_bank=build_anchor_bank)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(output_path))
        return payload


def _validate_artifact_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Artifact payload must be a dict.")
    if "metadata" not in payload or "samples" not in payload:
        raise ValueError("Artifact payload must contain metadata and samples.")
    if not isinstance(payload["samples"], list):
        raise ValueError("Artifact samples must be a list.")

    metadata = payload["metadata"]
    samples = payload["samples"]
    semantic_version = int(metadata.get("artifact_semantic_version", 1))
    if semantic_version != CURRENT_ARTIFACT_SEMANTIC_VERSION:
        raise ValueError(
            "Artifact semantic version "
            f"{semantic_version} is incompatible with current loader version "
            f"{CURRENT_ARTIFACT_SEMANTIC_VERSION}. Rebuild artifacts because polyline point xy semantics changed."
        )

    history_frames = int(metadata["history_frames"])
    future_frames = int(metadata["future_frames"])
    history_feature_dim = int(metadata["history_feature_dim"])
    polyline_point_dim = int(metadata["polyline_point_dim"])
    polyline_attr_dim = int(metadata["polyline_attr_dim"])
    object_feature_dim = int(metadata["object_feature_dim"])

    for sample_idx, sample in enumerate(samples):
        num_agents = int(sample["num_agents"])
        num_polylines = int(sample["num_polylines"])
        num_objects = int(sample["num_objects"])

        if len(sample["agent_tokens"]) != num_agents:
            raise ValueError(f"Sample {sample_idx} has inconsistent agent token count.")
        if sample["class_ids"].shape != (num_agents,):
            raise ValueError(f"Sample {sample_idx} has invalid class_ids shape.")
        if sample["history_features"].shape != (num_agents, history_frames, history_feature_dim):
            raise ValueError(f"Sample {sample_idx} has invalid history_features shape.")
        if sample["future_positions_ego"].shape != (num_agents, future_frames, 2):
            raise ValueError(f"Sample {sample_idx} has invalid future_positions_ego shape.")

        if sample["polyline_point_features"].ndim != 3:
            raise ValueError(f"Sample {sample_idx} has invalid polyline_point_features rank.")
        if sample["polyline_point_features"].shape[0] != num_polylines:
            raise ValueError(f"Sample {sample_idx} has inconsistent polyline count.")
        if sample["polyline_point_features"].shape[2] != polyline_point_dim:
            raise ValueError(f"Sample {sample_idx} has invalid polyline point feature dim.")
        if sample["polyline_point_mask"].shape != sample["polyline_point_features"].shape[:2]:
            raise ValueError(f"Sample {sample_idx} has invalid polyline_point_mask shape.")
        if sample["polyline_attrs"].shape != (num_polylines, polyline_attr_dim):
            raise ValueError(f"Sample {sample_idx} has invalid polyline_attrs shape.")

        if sample["object_features"].shape != (num_objects, object_feature_dim):
            raise ValueError(f"Sample {sample_idx} has invalid object_features shape.")

    anchor_bank = payload.get("anchor_bank")
    if anchor_bank is None:
        return
    if anchor_bank.get("method") != "kmeans":
        raise ValueError("Artifact anchor bank must be built with method='kmeans'. Rebuild artifacts.")
    if anchor_bank["full_bank"].ndim != 3 or anchor_bank["full_bank"].shape[-1] != 2:
        raise ValueError("Anchor bank full_bank must have shape [K, T, 2].")
    if anchor_bank["prefix_bank"].ndim != 3 or anchor_bank["prefix_bank"].shape[-1] != 2:
        raise ValueError("Anchor bank prefix_bank must have shape [K, T, 2].")


class V1ArtifactDataset(Dataset):
    def __init__(self, artifact_path: str | Path, augmentation: V1AugmentationConfig | None = None) -> None:
        self.artifact_path = str(artifact_path)
        self.augmentation = augmentation
        payload = torch.load(self.artifact_path, map_location="cpu", weights_only=False)
        _validate_artifact_payload(payload)
        self.samples = payload["samples"]
        self.metadata = payload["metadata"]
        self.anchor_bank = payload.get("anchor_bank")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return _apply_scene_augmentation(self.samples[idx], self.augmentation)


def _sample_future_local(sample: dict[str, Any]) -> np.ndarray:
    history = sample["history_features"].numpy()
    future = sample["future_positions_ego"].numpy()
    last_xy = history[:, -1, 0:2]
    cos_yaw = history[:, -1, 2]
    sin_yaw = history[:, -1, 3]
    yaw = np.arctan2(sin_yaw, cos_yaw).astype(np.float32, copy=False)
    rotation = np.stack(
        [
            np.stack([np.cos(yaw), np.sin(yaw)], axis=-1),
            np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32, copy=False)
    rel = future - last_xy[:, None, :]
    return np.matmul(rotation[:, None, :, :], rel[..., None]).squeeze(-1).astype(np.float32, copy=False)


def _direction_profile(future_local: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    displacement = float(np.linalg.norm(future_local[-1]))
    prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), future_local[:-1]], axis=0)
    delta = future_local - prev
    step_len = np.linalg.norm(delta, axis=-1)
    step_dir = delta / np.maximum(step_len[:, None], 1e-6)
    weights = step_len / max(float(step_len.sum()), 1e-6)
    return displacement, step_dir.astype(np.float32, copy=False), weights.astype(np.float32, copy=False)


def _normalize_profile(profile: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(profile, axis=-1, keepdims=True)
    return (profile / np.maximum(norm, 1e-6)).astype(np.float32, copy=False)


def _profile_assignment_distance(flat: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    assignments = distance.argmin(axis=1)
    closest = distance[np.arange(distance.shape[0]), assignments]
    return assignments.astype(np.int64, copy=False), closest.astype(np.float32, copy=False)


def _init_kmeans_plus_plus(flat: np.ndarray, num_centers: int, rng: np.random.Generator) -> np.ndarray:
    first_idx = int(rng.integers(0, flat.shape[0]))
    center_indices = [first_idx]
    closest = ((flat - flat[first_idx : first_idx + 1]) ** 2).sum(axis=-1)

    while len(center_indices) < num_centers:
        total = float(closest.sum())
        if total <= 1e-12:
            unused = np.setdiff1d(np.arange(flat.shape[0]), np.asarray(center_indices), assume_unique=False)
            if unused.size == 0:
                next_idx = int(rng.integers(0, flat.shape[0]))
            else:
                next_idx = int(rng.choice(unused))
        else:
            threshold = float(rng.random() * total)
            next_idx = int(np.searchsorted(np.cumsum(closest), threshold, side="right"))
            next_idx = min(next_idx, flat.shape[0] - 1)
        center_indices.append(next_idx)
        next_distance = ((flat - flat[next_idx : next_idx + 1]) ** 2).sum(axis=-1)
        closest = np.minimum(closest, next_distance)

    return np.asarray(center_indices, dtype=np.int64)


def _compute_r_max(
    future_local: np.ndarray,
    assignments: np.ndarray,
    anchors: np.ndarray,
) -> float:
    assigned = anchors[assignments]
    prev = np.concatenate([np.zeros((future_local.shape[0], 1, 2), dtype=np.float32), future_local[:, :-1, :]], axis=1)
    delta = future_local - prev
    progress = np.maximum((delta * assigned).sum(axis=-1), 0.0)
    base = np.cumsum(progress[..., None] * assigned, axis=1)
    tangent = assigned
    normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
    residual = future_local - base
    residual_t = (residual * tangent).sum(axis=-1)
    residual_n = (residual * normal).sum(axis=-1)
    residual_abs = np.abs(np.concatenate([residual_t.reshape(-1), residual_n.reshape(-1)], axis=0))
    return float(max(np.percentile(residual_abs, 95.0), 1e-3))


def build_anchor_bank_kmeans(samples: list[dict[str, Any]], cfg: V1DataConfig) -> dict[str, Any]:
    profiles: list[np.ndarray] = []
    future_local_list: list[np.ndarray] = []

    for sample in samples:
        future_local = _sample_future_local(sample)
        for agent_idx in range(future_local.shape[0]):
            displacement, profile, _ = _direction_profile(future_local[agent_idx])
            if displacement <= cfg.anchor_stationary_threshold_m:
                continue
            profiles.append(profile)
            future_local_list.append(future_local[agent_idx])

    if not profiles:
        raise ValueError("No directional trajectories available for anchor building.")

    profile_array = np.stack(profiles, axis=0).astype(np.float32, copy=False)
    future_local_array = np.stack(future_local_list, axis=0).astype(np.float32, copy=False)
    flat = profile_array.reshape(profile_array.shape[0], -1)

    rng = np.random.default_rng(cfg.anchor_random_seed)
    num_anchors = min(cfg.anchor_k12, flat.shape[0])
    center_idx = _init_kmeans_plus_plus(flat, num_anchors, rng)
    centers = profile_array[center_idx].copy()
    empty_replacements = 0
    actual_iterations = 0

    for iteration in range(max(int(cfg.anchor_kmeans_iters), 1)):
        actual_iterations = iteration + 1
        center_flat = centers.reshape(num_anchors, -1)
        assignments, closest = _profile_assignment_distance(flat, center_flat)
        next_centers = np.empty_like(centers)
        changed = False
        fallback_order = np.argsort(closest)[::-1]
        fallback_cursor = 0
        used_fallback: set[int] = set()

        for anchor_idx in range(num_anchors):
            member_mask = assignments == anchor_idx
            if np.any(member_mask):
                next_centers[anchor_idx] = _normalize_profile(profile_array[member_mask].mean(axis=0))
                continue

            while fallback_cursor < fallback_order.shape[0] and int(fallback_order[fallback_cursor]) in used_fallback:
                fallback_cursor += 1
            fallback_idx = int(fallback_order[min(fallback_cursor, fallback_order.shape[0] - 1)])
            used_fallback.add(fallback_idx)
            empty_replacements += 1
            changed = True
            next_centers[anchor_idx] = profile_array[fallback_idx]

        if np.allclose(centers, next_centers, atol=1e-6) and not changed:
            centers = next_centers
            break
        centers = next_centers

    full_bank = centers.astype(np.float32, copy=False)
    prefix_flat = np.round(full_bank[:, :6, :].reshape(full_bank.shape[0], -1), decimals=6)
    _, unique_idx = np.unique(prefix_flat, axis=0, return_index=True)
    prefix_bank = full_bank[np.sort(unique_idx), :6, :].astype(np.float32, copy=False)

    assignments, closest = _profile_assignment_distance(flat, full_bank.reshape(full_bank.shape[0], -1))
    r_max = _compute_r_max(future_local_array, assignments, full_bank)
    assignment_distance = np.sqrt(closest)

    return {
        "full_bank": full_bank,
        "prefix_bank": prefix_bank,
        "r_max": float(r_max),
        "method": "kmeans",
        "num_profiles": int(profile_array.shape[0]),
        "num_anchors": int(full_bank.shape[0]),
        "num_iterations": int(actual_iterations),
        "empty_cluster_replacements": int(empty_replacements),
        "mean_assignment_distance": float(assignment_distance.mean()),
        "max_assignment_distance": float(assignment_distance.max()),
        "seed": int(cfg.anchor_random_seed),
    }


def collate_v1_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_agents = max((int(item["num_agents"]) for item in batch), default=0)
    max_polylines = max((int(item["num_polylines"]) for item in batch), default=0)
    max_objects = max((int(item["num_objects"]) for item in batch), default=0)

    history_frames = batch[0]["history_features"].shape[1]
    history_dim = batch[0]["history_features"].shape[2]
    future_frames = batch[0]["future_positions_ego"].shape[1]
    max_polyline_points = batch[0]["polyline_point_features"].shape[1]
    polyline_attr_dim = batch[0]["polyline_attrs"].shape[1]
    object_feature_dim = batch[0]["object_features"].shape[1]

    packed: dict[str, Any] = {
        "class_ids": torch.full((batch_size, max_agents), -1, dtype=torch.long),
        "history_features": torch.zeros((batch_size, max_agents, history_frames, history_dim), dtype=torch.float32),
        "future_positions_ego": torch.zeros((batch_size, max_agents, future_frames, 2), dtype=torch.float32),
        "agent_pad_mask": torch.ones((batch_size, max_agents), dtype=torch.bool),
        "polyline_point_features": torch.zeros((batch_size, max_polylines, max_polyline_points, 4), dtype=torch.float32),
        "polyline_point_mask": torch.zeros((batch_size, max_polylines, max_polyline_points), dtype=torch.bool),
        "polyline_attrs": torch.zeros((batch_size, max_polylines, polyline_attr_dim), dtype=torch.float32),
        "polyline_pad_mask": torch.ones((batch_size, max_polylines), dtype=torch.bool),
        "object_features": torch.zeros((batch_size, max_objects, object_feature_dim), dtype=torch.float32),
        "object_pad_mask": torch.ones((batch_size, max_objects), dtype=torch.bool),
    }

    scene_tokens: list[str] = []
    sample_tokens: list[str] = []
    map_names: list[str] = []
    agent_tokens: list[list[str]] = []
    window_start_idx: list[int] = []

    for batch_idx, item in enumerate(batch):
        num_agents = int(item["num_agents"])
        num_polylines = int(item["num_polylines"])
        num_objects = int(item["num_objects"])

        scene_tokens.append(str(item["scene_token"]))
        sample_tokens.append(str(item["sample_token"]))
        map_names.append(str(item["map_name"]))
        agent_tokens.append(list(item["agent_tokens"]))
        window_start_idx.append(int(item["window_start_idx"]))

        packed["agent_pad_mask"][batch_idx, :num_agents] = False
        packed["class_ids"][batch_idx, :num_agents] = item["class_ids"]
        packed["history_features"][batch_idx, :num_agents] = item["history_features"]
        packed["future_positions_ego"][batch_idx, :num_agents] = item["future_positions_ego"]

        packed["polyline_pad_mask"][batch_idx, :num_polylines] = False
        packed["polyline_point_features"][batch_idx, :num_polylines] = item["polyline_point_features"]
        packed["polyline_point_mask"][batch_idx, :num_polylines] = item["polyline_point_mask"]
        packed["polyline_attrs"][batch_idx, :num_polylines] = item["polyline_attrs"]

        packed["object_pad_mask"][batch_idx, :num_objects] = False
        packed["object_features"][batch_idx, :num_objects] = item["object_features"]

    packed["scene_tokens"] = scene_tokens
    packed["sample_tokens"] = sample_tokens
    packed["map_names"] = map_names
    packed["agent_tokens"] = agent_tokens
    packed["window_start_idx"] = window_start_idx
    return packed


def build_v1_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_v1_batch,
        "persistent_workers": num_workers > 0,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)
