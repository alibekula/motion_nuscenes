from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.motion_dataset import MotionDatasetConfig, MotionPredictionDataset
from data.preprocessed_dataset import PreprocessedDatasetMetadata
from motion_v1.geometry import global_xy_to_ego, wrap_angle


def _normalize_direction_profile(profile: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(profile, axis=-1, keepdims=True)
    return profile / np.maximum(norm, eps)


def build_prefix_bank(full_bank: np.ndarray, decimals: int = 6) -> np.ndarray:
    if full_bank.ndim != 3 or full_bank.shape[1] < 6 or full_bank.shape[2] != 2:
        raise ValueError("Expected full_bank with shape [K12, >=6, 2].")

    prefixes = _normalize_direction_profile(np.asarray(full_bank[:, :6, :], dtype=np.float32))
    rounded = np.round(prefixes.reshape(prefixes.shape[0], -1), decimals=decimals)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return prefixes[np.sort(unique_indices)]


@dataclass(frozen=True)
class AnchorBank:
    full_bank: np.ndarray
    prefix_bank: np.ndarray


@dataclass(frozen=True)
class OfflinePreprocessingConfig:
    map_crop_radius_m: float = 80.0
    polyline_resolution_m: float = 1.0
    max_polyline_points: int = 20
    anchor_k12: int = 64
    anchor_kmeans_iters: int = 50
    anchor_random_seed: int = 13
    anchor_stationary_disp_threshold_m: float = 0.5
    nearest_edges_per_object: int = 3
    include_divider_layer: bool = True
    polyline_point_dim: int = 4
    polyline_attr_dim: int = 8
    object_feature_dim: int = 10
    edge_attr_dim: int = 12


@dataclass(frozen=True)
class AnchorCoverageReport:
    num_profiles: int
    num_stationary_skipped: int
    mean_cosine_distance: float
    p95_cosine_distance: float
    anchor_usage_hist: tuple[int, ...]
    class_usage_hist: dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class AnchorResidualStats:
    r_max: float
    residual_abs_p95: float
    residual_abs_mean: float
    mean_progress: float


@dataclass(frozen=True)
class PolylineMapRecord:
    token: str
    polyline_type: str
    points_global: np.ndarray
    attrs: np.ndarray
    connected_count: int
    is_fork: bool
    width: float
    geometry: Any | None = None


@dataclass(frozen=True)
class ObjectMapRecord:
    token: str
    object_type: str
    centroid_global: np.ndarray
    yaw_global: float
    size_xy: np.ndarray
    associated_polyline_tokens: tuple[str, ...]
    geometry: Any | None = None


@dataclass(frozen=True)
class ObjectPolylineEdgeRecord:
    object_token: str
    polyline_token: str
    edge_attr: np.ndarray
    edge_type: str


@dataclass(frozen=True)
class AnchorBankBuildReport:
    coverage: AnchorCoverageReport
    residual_stats: AnchorResidualStats

    @property
    def r_max(self) -> float:
        return float(self.residual_stats.r_max)

    @property
    def residual_abs_p95(self) -> float:
        return float(self.residual_stats.residual_abs_p95)

    @property
    def residual_abs_mean(self) -> float:
        return float(self.residual_stats.residual_abs_mean)

    @property
    def mean_progress(self) -> float:
        return float(self.residual_stats.mean_progress)


class OfflineMotionPreprocessor:
    """
    Offline preprocessing pipeline.

    Responsibilities:
    - materialize per-sample agent tensors
    - extract cropped map polylines and map objects in ego frame
    - precompute a static object-polyline edge table per map and filter it per sample crop
    - accumulate direction profiles in the same pass to build A12, then A6
    - estimate r_max from nearest-anchor residual statistics on the train split
    """

    def __init__(
        self,
        nusc,
        dataset_cfg: MotionDatasetConfig | None = None,
        preprocess_cfg: OfflinePreprocessingConfig | None = None,
    ) -> None:
        self.nusc = nusc
        self.dataset_cfg = dataset_cfg or MotionDatasetConfig()
        self.preprocess_cfg = preprocess_cfg or OfflinePreprocessingConfig()
        self._map_cache_by_location: dict[str, dict[str, list[Any]]] = {}

    def preprocess_split(
        self,
        *,
        split_name: str,
        sample_tokens: list[str],
        build_anchor_bank_flag: bool,
    ) -> dict[str, Any]:
        raw_dataset = MotionPredictionDataset(self.nusc, cfg=self.dataset_cfg, sample_tokens=sample_tokens)

        samples: list[dict[str, Any]] = []
        anchor_profiles: list[np.ndarray] = []
        anchor_profile_weights: list[np.ndarray] = []
        anchor_future_local: list[np.ndarray] = []
        anchor_profile_class_ids: list[int] = []
        stationary_skipped = 0

        for idx in range(len(raw_dataset)):
            raw_item = raw_dataset[idx]
            sample = self._materialize_sample(raw_dataset, raw_item)
            samples.append(sample)

            if not build_anchor_bank_flag:
                continue

            gt_local = self._future_positions_local(sample)
            train_valid = sample["train_valid_mask"].numpy()
            class_ids = sample["class_ids"].numpy()
            for agent_idx in range(sample["num_agents"]):
                if not bool(train_valid[agent_idx]):
                    continue
                directional = self._direction_profile_or_none(gt_local[agent_idx])
                if directional is None:
                    stationary_skipped += 1
                    continue
                profile, weights = directional
                anchor_profiles.append(profile)
                anchor_profile_weights.append(weights)
                anchor_future_local.append(gt_local[agent_idx].astype(np.float32, copy=False))
                anchor_profile_class_ids.append(int(class_ids[agent_idx]))

        metadata = PreprocessedDatasetMetadata(
            split_name=split_name,
            history_frames=self.dataset_cfg.history_frames,
            future_frames=self.dataset_cfg.future_frames,
            history_feature_dim=11 + len(self.dataset_cfg.class_names),
            polyline_point_dim=self.preprocess_cfg.polyline_point_dim,
            polyline_attr_dim=self.preprocess_cfg.polyline_attr_dim,
            object_feature_dim=self.preprocess_cfg.object_feature_dim,
            edge_attr_dim=self.preprocess_cfg.edge_attr_dim,
        )

        payload: dict[str, Any] = {
            "metadata": asdict(metadata),
            "samples": samples,
        }
        if build_anchor_bank_flag:
            anchor_bank, coverage = self._build_anchor_bank(
                anchor_profiles,
                anchor_profile_weights,
                anchor_future_local,
                anchor_profile_class_ids,
                stationary_skipped,
            )
            payload["anchor_bank"] = {
                "full_bank": anchor_bank.full_bank,
                "prefix_bank": anchor_bank.prefix_bank,
                "r_max": float(coverage.r_max),
                "residual_stats": {
                    "r_max": float(coverage.r_max),
                    "residual_abs_p95": float(coverage.residual_abs_p95),
                    "residual_abs_mean": float(coverage.residual_abs_mean),
                    "mean_progress": float(coverage.mean_progress),
                },
            }
            payload["coverage_report"] = asdict(coverage.coverage)
        return payload

    def save_split_artifacts(
        self,
        output_path: str | Path,
        *,
        split_name: str,
        sample_tokens: list[str],
        build_anchor_bank_flag: bool,
    ) -> dict[str, Any]:
        payload = self.preprocess_split(
            split_name=split_name,
            sample_tokens=sample_tokens,
            build_anchor_bank_flag=build_anchor_bank_flag,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(output_path))
        return payload

    def _materialize_sample(
        self,
        raw_dataset: MotionPredictionDataset,
        raw_item: dict[str, Any],
    ) -> dict[str, Any]:
        sample_token = str(raw_item["sample_token"])
        sample = raw_dataset._get_sample(sample_token)
        scene = self.nusc.get("scene", str(sample["scene_token"]))
        log = self.nusc.get("log", str(scene["log_token"]))
        map_name = str(log["location"])
        ego_xy_global, ego_yaw_global = raw_dataset._get_ego_pose(sample_token)

        map_features = self._extract_map_features_for_sample(
            map_name=map_name,
            ego_xy_global=ego_xy_global,
            ego_yaw_global=ego_yaw_global,
        )

        materialized: dict[str, Any] = {}
        for key, value in raw_item.items():
            if isinstance(value, torch.Tensor):
                materialized[key] = value.detach().cpu().clone()
            elif isinstance(value, list):
                materialized[key] = list(value)
            else:
                materialized[key] = value

        materialized.update(
            {
                "map_name": map_name,
                "num_polylines": int(map_features["num_polylines"]),
                "num_polyline_points": int(map_features["num_polyline_points"]),
                "num_objects": int(map_features["num_objects"]),
                "num_edges": int(map_features["num_edges"]),
                "polyline_point_features": map_features["polyline_point_features"],
                "polyline_point_mask": map_features["polyline_point_mask"],
                "polyline_attrs": map_features["polyline_attrs"],
                "polyline_tokens": list(map_features["polyline_tokens"]),
                "object_features": map_features["object_features"],
                "object_tokens": list(map_features["object_tokens"]),
                "object_polyline_edge_index": map_features["object_polyline_edge_index"],
                "object_polyline_edge_attr": map_features["object_polyline_edge_attr"],
                "edge_types": list(map_features["edge_types"]),
            }
        )
        return materialized

    def _future_positions_local(self, sample: dict[str, Any]) -> np.ndarray:
        rel = sample["future_positions_ego"].numpy() - sample["p_last_hist_ego"].numpy()[:, None, :]
        rotation = sample["rotation_ego_to_local"].numpy()[:, None, :, :]
        return np.matmul(rotation, rel[..., None]).squeeze(-1)

    def _direction_profile_or_none(self, future_local: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if future_local.shape != (self.dataset_cfg.future_frames, 2):
            raise ValueError("Expected future_local with shape [future_frames, 2].")
        displacement = float(np.linalg.norm(future_local[-1]))
        if displacement <= self.preprocess_cfg.anchor_stationary_disp_threshold_m:
            return None

        prev = np.concatenate([np.zeros((1, 2), dtype=np.float32), future_local[:-1]], axis=0)
        delta = future_local - prev
        step_len = np.linalg.norm(delta, axis=-1)
        step_norm = np.maximum(step_len[:, None], 1e-6)
        weights = (step_len / max(float(step_len.sum()), 1e-6)).astype(np.float32, copy=False)
        profile = (delta / step_norm).astype(np.float32, copy=False)
        return profile, weights

    def _build_anchor_bank(
        self,
        profiles: list[np.ndarray],
        profile_weights: list[np.ndarray],
        future_local_trajectories: list[np.ndarray],
        class_ids: list[int],
        stationary_skipped: int,
    ) -> tuple[AnchorBank, AnchorBankBuildReport]:
        if not profiles:
            raise ValueError("No directional training profiles collected; cannot build anchor bank.")

        profile_array = np.stack(profiles, axis=0).astype(np.float32, copy=False)
        weight_array = np.stack(profile_weights, axis=0).astype(np.float32, copy=False)
        future_local_array = np.stack(future_local_trajectories, axis=0).astype(np.float32, copy=False)
        flat = profile_array.reshape(profile_array.shape[0], -1)
        centroids = _kmeans_l2(
            flat,
            k=min(self.preprocess_cfg.anchor_k12, flat.shape[0]),
            num_iters=self.preprocess_cfg.anchor_kmeans_iters,
            seed=self.preprocess_cfg.anchor_random_seed,
        ).reshape(-1, self.dataset_cfg.future_frames, 2)
        full_bank = _normalize_directions(centroids)
        prefix_bank = build_prefix_bank(full_bank)

        anchor_bank = AnchorBank(
            full_bank=full_bank.astype(np.float32, copy=False),
            prefix_bank=prefix_bank.astype(np.float32, copy=False),
        )

        distances = _weighted_cosine_distances(profile_array, weight_array, full_bank)
        nearest = distances.argmin(axis=1)
        nearest_distance = distances[np.arange(distances.shape[0]), nearest]

        class_usage_hist: dict[int, tuple[int, ...]] = {}
        class_id_array = np.asarray(class_ids, dtype=np.int64)
        for class_id in np.unique(class_id_array):
            hist = np.bincount(nearest[class_id_array == class_id], minlength=full_bank.shape[0])
            class_usage_hist[int(class_id)] = tuple(int(v) for v in hist.tolist())

        coverage = AnchorCoverageReport(
            num_profiles=int(profile_array.shape[0]),
            num_stationary_skipped=int(stationary_skipped),
            mean_cosine_distance=float(nearest_distance.mean()),
            p95_cosine_distance=float(np.percentile(nearest_distance, 95.0)),
            anchor_usage_hist=tuple(int(v) for v in np.bincount(nearest, minlength=full_bank.shape[0]).tolist()),
            class_usage_hist=class_usage_hist,
        )
        residual_stats = _compute_r_max_from_assignments(
            future_local_trajectories=future_local_array,
            anchor_assignments=nearest,
            anchor_bank=full_bank,
        )
        return anchor_bank, AnchorBankBuildReport(coverage=coverage, residual_stats=residual_stats)

    def _extract_map_features_for_sample(
        self,
        *,
        map_name: str,
        ego_xy_global: np.ndarray,
        ego_yaw_global: float,
    ) -> dict[str, Any]:
        crop_radius = float(self.preprocess_cfg.map_crop_radius_m)
        crop_records = self._get_or_build_map_cache(map_name)
        crop_center = np.asarray(ego_xy_global, dtype=np.float32)

        polyline_entries: list[dict[str, Any]] = []
        polyline_token_to_idx: dict[str, int] = {}
        for record in crop_records["polylines"]:
            points_global = record.points_global
            if points_global.shape[0] < 2:
                continue
            if np.linalg.norm(points_global - crop_center[None, :], axis=-1).min() > crop_radius:
                continue
            local = self._polyline_points_to_features(points_global, crop_center, ego_yaw_global)
            if local is None:
                continue
            polyline_token_to_idx[record.token] = len(polyline_entries)
            polyline_entries.append(
                {
                    "token": record.token,
                    "type": record.polyline_type,
                    "points_global": points_global,
                    "geometry": record.geometry,
                    "point_features": local["point_features"],
                    "point_mask": local["point_mask"],
                    "attrs": record.attrs.astype(np.float32, copy=False),
                }
            )

        object_entries: list[dict[str, Any]] = []
        object_token_to_idx: dict[str, int] = {}
        for record in crop_records["objects"]:
            distance = float(np.linalg.norm(record.centroid_global - crop_center))
            if distance > crop_radius:
                continue
            object_token_to_idx[record.token] = len(object_entries)
            object_entries.append(
                {
                    "token": record.token,
                    "type": record.object_type,
                    "centroid_global": record.centroid_global,
                    "yaw_global": record.yaw_global,
                    "size_xy": record.size_xy,
                    "associated_polyline_tokens": record.associated_polyline_tokens,
                    "geometry": record.geometry,
                    "features": self._object_to_features(record, crop_center, ego_yaw_global),
                }
            )

        edge_index: list[list[int]] = []
        edge_attr: list[np.ndarray] = []
        edge_types: list[str] = []
        for edge_record in crop_records["edges"]:
            obj_idx = object_token_to_idx.get(edge_record.object_token)
            poly_idx = polyline_token_to_idx.get(edge_record.polyline_token)
            if obj_idx is None or poly_idx is None:
                continue
            edge_index.append([obj_idx, poly_idx])
            edge_attr.append(edge_record.edge_attr.astype(np.float32, copy=False))
            edge_types.append(edge_record.edge_type)

        num_polylines = len(polyline_entries)
        max_points = self.preprocess_cfg.max_polyline_points
        polyline_point_features = torch.zeros((num_polylines, max_points, self.preprocess_cfg.polyline_point_dim), dtype=torch.float32)
        polyline_point_mask = torch.zeros((num_polylines, max_points), dtype=torch.bool)
        polyline_attrs = torch.zeros((num_polylines, self.preprocess_cfg.polyline_attr_dim), dtype=torch.float32)
        for poly_idx, entry in enumerate(polyline_entries):
            polyline_point_features[poly_idx] = torch.from_numpy(entry["point_features"])
            polyline_point_mask[poly_idx] = torch.from_numpy(entry["point_mask"])
            polyline_attrs[poly_idx] = torch.from_numpy(entry["attrs"])

        num_objects = len(object_entries)
        object_features = torch.zeros((num_objects, self.preprocess_cfg.object_feature_dim), dtype=torch.float32)
        for obj_idx, entry in enumerate(object_entries):
            object_features[obj_idx] = torch.from_numpy(entry["features"])

        num_edges = len(edge_index)
        edge_index_tensor = torch.zeros((num_edges, 2), dtype=torch.long)
        edge_attr_tensor = torch.zeros((num_edges, self.preprocess_cfg.edge_attr_dim), dtype=torch.float32)
        for edge_idx, value in enumerate(edge_index):
            edge_index_tensor[edge_idx] = torch.tensor(value, dtype=torch.long)
            edge_attr_tensor[edge_idx] = torch.from_numpy(edge_attr[edge_idx])

        return {
            "num_polylines": num_polylines,
            "num_polyline_points": max_points,
            "num_objects": num_objects,
            "num_edges": num_edges,
            "polyline_point_features": polyline_point_features,
            "polyline_point_mask": polyline_point_mask,
            "polyline_attrs": polyline_attrs,
            "polyline_tokens": [entry["token"] for entry in polyline_entries],
            "object_features": object_features,
            "object_tokens": [entry["token"] for entry in object_entries],
            "object_polyline_edge_index": edge_index_tensor,
            "object_polyline_edge_attr": edge_attr_tensor,
            "edge_types": edge_types,
        }

    def _get_or_build_map_cache(self, map_name: str) -> dict[str, list[Any]]:
        cached = self._map_cache_by_location.get(map_name)
        if cached is not None:
            return cached

        try:
            from shapely.geometry import LineString
            from nuscenes.map_expansion.map_api import NuScenesMap
        except ImportError as exc:
            raise ImportError(
                "Offline map preprocessing requires nuscenes.map_expansion.map_api."
            ) from exc

        nusc_map = NuScenesMap(dataroot=self.nusc.dataroot, map_name=map_name)
        polylines: list[PolylineMapRecord] = []
        objects: list[ObjectMapRecord] = []

        lane_layers = [("lane", "lane"), ("lane_connector", "lane_connector")]
        for layer_name, poly_type in lane_layers:
            layer_records = getattr(nusc_map, layer_name, [])
            tokens = [str(record["token"]) for record in layer_records]
            discretized = nusc_map.discretize_lanes(tokens, self.preprocess_cfg.polyline_resolution_m) if tokens else {}
            for record in layer_records:
                token = str(record["token"])
                points = _as_xy_array(discretized.get(token, []))
                if points.shape[0] < 2:
                    continue
                incoming = _safe_lane_ids(nusc_map, "get_incoming_lane_ids", token)
                outgoing = _safe_lane_ids(nusc_map, "get_outgoing_lane_ids", token)
                connected_count = len(incoming) + len(outgoing)
                is_fork = len(outgoing) > 1 or len(incoming) > 1
                width = float(record.get("width", 0.0) or 0.0)
                attrs = self._build_polyline_attrs(poly_type, points, width, connected_count, is_fork)
                polylines.append(
                    PolylineMapRecord(
                        token=token,
                        polyline_type=poly_type,
                        points_global=points.astype(np.float32, copy=False),
                        attrs=attrs,
                        connected_count=connected_count,
                        is_fork=is_fork,
                        width=width,
                        geometry=LineString(points),
                    )
                )

        if self.preprocess_cfg.include_divider_layer:
            for layer_name in ("road_divider", "lane_divider"):
                layer_records = getattr(nusc_map, layer_name, [])
                for record in layer_records:
                    line = _extract_line_from_record(nusc_map, record)
                    if line is None:
                        continue
                    points = _as_xy_array(line.coords)
                    if points.shape[0] < 2:
                        continue
                    attrs = self._build_polyline_attrs("divider", points, 0.0, 0, False)
                    polylines.append(
                        PolylineMapRecord(
                            token=str(record["token"]),
                            polyline_type="divider",
                            points_global=points.astype(np.float32, copy=False),
                            attrs=attrs,
                            connected_count=0,
                            is_fork=False,
                            width=0.0,
                            geometry=LineString(points),
                        )
                    )

        object_layers = {
            "ped_crossing": "crosswalk",
            "stop_line": "stop_line",
            "traffic_light": "traffic_light",
            "carpark_area": "carpark",
        }
        for layer_name, object_type in object_layers.items():
            layer_records = getattr(nusc_map, layer_name, [])
            for record in layer_records:
                geometry = _extract_geometry_from_record(nusc_map, record)
                if geometry is None:
                    continue
                centroid_global, yaw_global, size_xy = _geometry_pose_and_size(geometry)
                objects.append(
                    ObjectMapRecord(
                        token=str(record["token"]),
                        object_type=object_type,
                        centroid_global=centroid_global.astype(np.float32, copy=False),
                        yaw_global=float(yaw_global),
                        size_xy=size_xy.astype(np.float32, copy=False),
                        associated_polyline_tokens=_extract_associated_polyline_tokens(record),
                        geometry=geometry,
                        )
                    )

        polyline_token_to_idx = {record.token: idx for idx, record in enumerate(polylines)}
        edge_records: list[ObjectPolylineEdgeRecord] = []
        for object_record in objects:
            for poly_idx, attr, edge_type in self._build_edges_for_object(
                obj_entry={
                    "token": object_record.token,
                    "type": object_record.object_type,
                    "centroid_global": object_record.centroid_global,
                    "yaw_global": object_record.yaw_global,
                    "associated_polyline_tokens": object_record.associated_polyline_tokens,
                    "geometry": object_record.geometry,
                },
                obj_idx=0,
                polyline_entries=[
                    {
                        "token": poly_record.token,
                        "points_global": poly_record.points_global,
                        "geometry": poly_record.geometry,
                    }
                    for poly_record in polylines
                ],
                polyline_token_to_idx=polyline_token_to_idx,
            ):
                edge_records.append(
                    ObjectPolylineEdgeRecord(
                        object_token=object_record.token,
                        polyline_token=polylines[poly_idx].token,
                        edge_attr=attr.astype(np.float32, copy=False),
                        edge_type=edge_type,
                    )
                )

        self._map_cache_by_location[map_name] = {
            "polylines": polylines,
            "objects": objects,
            "edges": edge_records,
        }
        return self._map_cache_by_location[map_name]

    def _build_polyline_attrs(
        self,
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
        segment = np.diff(points_global, axis=0)
        length = float(np.linalg.norm(segment, axis=-1).sum())
        headings = np.arctan2(segment[:, 1], segment[:, 0]) if segment.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
        heading_delta = wrap_angle(headings[1:] - headings[:-1]) if headings.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
        curvature = float(np.abs(heading_delta).sum() / max(length, 1e-6))
        return np.concatenate(
            [
                type_onehot,
                np.asarray(
                    [
                        width,
                        length,
                        curvature,
                        float(connected_count),
                        float(is_fork),
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

    def _polyline_points_to_features(
        self,
        points_global: np.ndarray,
        ego_xy_global: np.ndarray,
        ego_yaw_global: float,
    ) -> dict[str, np.ndarray] | None:
        local_points = np.stack(
            [global_xy_to_ego(point, ego_xy_global, ego_yaw_global) for point in points_global],
            axis=0,
        ).astype(np.float32, copy=False)
        if local_points.shape[0] < 2:
            return None
        resampled = _resample_polyline(local_points, self.preprocess_cfg.max_polyline_points)
        if resampled.shape[0] < 2:
            return None

        num_points = resampled.shape[0]
        point_mask = np.zeros((self.preprocess_cfg.max_polyline_points,), dtype=np.bool_)
        point_mask[:num_points] = True

        midpoint = resampled[num_points // 2]
        tangent = np.zeros_like(resampled)
        if num_points > 1:
            tangent[1:-1] = 0.5 * (resampled[2:] - resampled[:-2])
            tangent[0] = resampled[1] - resampled[0]
            tangent[-1] = resampled[-1] - resampled[-2]

        point_features = np.zeros((self.preprocess_cfg.max_polyline_points, self.preprocess_cfg.polyline_point_dim), dtype=np.float32)
        point_features[:num_points, 0:2] = resampled - midpoint[None, :]
        point_features[:num_points, 2:4] = tangent
        return {
            "point_features": point_features,
            "point_mask": point_mask,
        }

    def _object_to_features(
        self,
        record: ObjectMapRecord,
        ego_xy_global: np.ndarray,
        ego_yaw_global: float,
    ) -> np.ndarray:
        xy_ego = global_xy_to_ego(record.centroid_global, ego_xy_global, ego_yaw_global)
        yaw_ego = float(wrap_angle(record.yaw_global - ego_yaw_global))
        type_onehot = {
            "crosswalk": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "stop_line": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "traffic_light": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "carpark": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }[record.object_type]
        return np.concatenate(
            [
                xy_ego.astype(np.float32, copy=False),
                np.asarray([np.cos(yaw_ego), np.sin(yaw_ego)], dtype=np.float32),
                record.size_xy.astype(np.float32, copy=False),
                type_onehot,
            ],
            axis=0,
        )

    def _build_edges_for_object(
        self,
        *,
        obj_entry: dict[str, Any],
        obj_idx: int,
        polyline_entries: list[dict[str, Any]],
        polyline_token_to_idx: dict[str, int],
    ) -> list[tuple[int, np.ndarray, str]]:
        del obj_idx

        candidate_edges: list[tuple[int, str]] = []
        for token in obj_entry["associated_polyline_tokens"]:
            poly_idx = polyline_token_to_idx.get(token)
            if poly_idx is None:
                continue
            candidate_edges.append((poly_idx, "associated"))

        if not candidate_edges and obj_entry["geometry"] is not None:
            for poly_idx, poly in enumerate(polyline_entries):
                geometry = poly.get("geometry")
                if geometry is None:
                    continue
                try:
                    if obj_entry["geometry"].intersects(geometry):
                        candidate_edges.append((poly_idx, "intersects"))
                except Exception:
                    continue

        if not candidate_edges:
            nearest: list[tuple[float, int]] = []
            for poly_idx, poly in enumerate(polyline_entries):
                distance, _, _ = _nearest_polyline_measure(poly["points_global"], obj_entry["centroid_global"])
                nearest.append((distance, poly_idx))
            nearest.sort(key=lambda item: item[0])
            for _, poly_idx in nearest[: self.preprocess_cfg.nearest_edges_per_object]:
                candidate_edges.append((poly_idx, "nearest"))

        dedup: dict[int, str] = {}
        for poly_idx, edge_type in candidate_edges:
            current = dedup.get(poly_idx)
            if current == "associated":
                continue
            if current == "intersects" and edge_type == "nearest":
                continue
            dedup[poly_idx] = edge_type if current is None or edge_type in {"associated", "intersects"} else current

        output: list[tuple[int, np.ndarray, str]] = []
        for poly_idx, edge_type in dedup.items():
            poly = polyline_entries[poly_idx]
            edge_feature = _build_edge_attr(
                polyline_points_global=poly["points_global"],
                object_centroid_global=obj_entry["centroid_global"],
                object_yaw_global=float(obj_entry["yaw_global"]),
                object_type=obj_entry["type"],
                edge_type=edge_type,
            )
            output.append((poly_idx, edge_feature, edge_type))
        return output


def _normalize_directions(profile: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(profile, axis=-1, keepdims=True)
    return profile / np.maximum(norm, 1e-6)


def _kmeans_l2(data: np.ndarray, k: int, num_iters: int, seed: int) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("Expected data with shape [N, D].")
    if data.shape[0] == 0:
        raise ValueError("Cannot run k-means on an empty dataset.")

    rng = np.random.default_rng(seed)
    initial_idx = rng.choice(data.shape[0], size=k, replace=False)
    centroids = data[initial_idx].copy()
    assignments = np.full((data.shape[0],), -1, dtype=np.int64)

    for _ in range(num_iters):
        distances = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        new_assignments = distances.argmin(axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for centroid_idx in range(k):
            members = data[assignments == centroid_idx]
            if members.shape[0] == 0:
                centroids[centroid_idx] = data[rng.integers(0, data.shape[0])]
            else:
                centroids[centroid_idx] = members.mean(axis=0)
    return centroids


def _weighted_cosine_distances(
    profiles: np.ndarray,
    weights: np.ndarray,
    anchor_bank: np.ndarray,
) -> np.ndarray:
    dot = (profiles[:, None, :, :] * anchor_bank[None, :, :, :]).sum(axis=-1)
    return (weights[:, None, :] * (1.0 - dot)).sum(axis=-1)


def _compute_r_max_from_assignments(
    future_local_trajectories: np.ndarray,
    anchor_assignments: np.ndarray,
    anchor_bank: np.ndarray,
) -> AnchorResidualStats:
    if future_local_trajectories.ndim != 3 or future_local_trajectories.shape[-1] != 2:
        raise ValueError("Expected future_local_trajectories with shape [N, H, 2].")
    if anchor_assignments.ndim != 1 or anchor_assignments.shape[0] != future_local_trajectories.shape[0]:
        raise ValueError("anchor_assignments must have shape [N].")
    if anchor_bank.ndim != 3 or anchor_bank.shape[-1] != 2:
        raise ValueError("Expected anchor_bank with shape [K, H, 2].")

    prev = np.concatenate(
        [np.zeros((future_local_trajectories.shape[0], 1, 2), dtype=np.float32), future_local_trajectories[:, :-1, :]],
        axis=1,
    )
    delta = future_local_trajectories - prev
    assigned_anchors = anchor_bank[anchor_assignments]
    progress = np.maximum((delta * assigned_anchors).sum(axis=-1), 0.0)
    base = np.cumsum(progress[..., None] * assigned_anchors, axis=1)

    tangent = assigned_anchors
    normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
    residual = future_local_trajectories - base
    residual_tangent = (residual * tangent).sum(axis=-1)
    residual_normal = (residual * normal).sum(axis=-1)
    residual_abs = np.abs(np.concatenate([residual_tangent.reshape(-1), residual_normal.reshape(-1)], axis=0))

    if residual_abs.size == 0:
        r_max = 1.0
        residual_abs_mean = 0.0
        residual_abs_p95 = 0.0
    else:
        residual_abs_mean = float(residual_abs.mean())
        residual_abs_p95 = float(np.percentile(residual_abs, 95.0))
        r_max = max(residual_abs_p95, 1e-3)

    return AnchorResidualStats(
        r_max=float(r_max),
        residual_abs_p95=float(residual_abs_p95),
        residual_abs_mean=float(residual_abs_mean),
        mean_progress=float(progress.mean()) if progress.size > 0 else 0.0,
    )


def _as_xy_array(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] >= 2:
        return array[:, :2].astype(np.float32, copy=False)
    return np.zeros((0, 2), dtype=np.float32)


def _safe_lane_ids(nusc_map, fn_name: str, token: str) -> list[str]:
    fn = getattr(nusc_map, fn_name, None)
    if fn is None:
        return []
    try:
        return list(fn(token))
    except Exception:
        return []


def _extract_line_from_record(nusc_map, record: dict[str, Any]):
    line_token = record.get("line_token")
    if line_token:
        try:
            return nusc_map.extract_line(str(line_token))
        except Exception:
            return None
    return None


def _extract_geometry_from_record(nusc_map, record: dict[str, Any]):
    polygon_token = record.get("polygon_token")
    if polygon_token:
        try:
            return nusc_map.extract_polygon(str(polygon_token))
        except Exception:
            pass

    polygon_tokens = record.get("polygon_tokens")
    if polygon_tokens:
        for token in polygon_tokens:
            try:
                geometry = nusc_map.extract_polygon(str(token))
                if geometry is not None:
                    return geometry
            except Exception:
                continue

    line = _extract_line_from_record(nusc_map, record)
    if line is not None:
        return line
    return None


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
        coords = _as_xy_array(geometry.coords)
        centroid = coords.mean(axis=0)
        delta = coords[-1] - coords[0]
        yaw = float(np.arctan2(delta[1], delta[0]))
        size_xy = np.asarray([0.5, max(float(np.linalg.norm(delta)), 0.5)], dtype=np.float32)
        return centroid, yaw, size_xy

    centroid = np.asarray([0.0, 0.0], dtype=np.float32)
    return centroid, 0.0, np.asarray([1.0, 1.0], dtype=np.float32)


def _extract_associated_polyline_tokens(record: dict[str, Any]) -> tuple[str, ...]:
    tokens: set[str] = set()
    for key, value in record.items():
        key_lower = str(key).lower()
        if "lane" not in key_lower and "connector" not in key_lower:
            continue
        if isinstance(value, str):
            tokens.add(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    tokens.add(item)
    return tuple(sorted(tokens))


def _resample_polyline(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points.astype(np.float32, copy=False)
    segment = np.diff(points, axis=0)
    seg_len = np.linalg.norm(segment, axis=-1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(arc[-1])
    if total < 1e-6:
        return points[:1].astype(np.float32, copy=False)
    target = np.linspace(0.0, total, num=max_points, dtype=np.float32)
    x = np.interp(target, arc, points[:, 0]).astype(np.float32)
    y = np.interp(target, arc, points[:, 1]).astype(np.float32)
    return np.stack([x, y], axis=-1)


def _nearest_polyline_measure(
    polyline_points_global: np.ndarray,
    object_centroid_global: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    if polyline_points_global.shape[0] < 2:
        return float("inf"), polyline_points_global[0], np.asarray([1.0, 0.0], dtype=np.float32)

    best_distance = float("inf")
    best_point = polyline_points_global[0]
    best_tangent = polyline_points_global[1] - polyline_points_global[0]

    for seg_idx in range(polyline_points_global.shape[0] - 1):
        p0 = polyline_points_global[seg_idx]
        p1 = polyline_points_global[seg_idx + 1]
        segment = p1 - p0
        denom = float(np.dot(segment, segment))
        if denom < 1e-6:
            continue
        alpha = float(np.dot(object_centroid_global - p0, segment) / denom)
        alpha = min(max(alpha, 0.0), 1.0)
        closest = p0 + alpha * segment
        distance = float(np.linalg.norm(object_centroid_global - closest))
        if distance < best_distance:
            best_distance = distance
            best_point = closest
            best_tangent = segment

    tangent_norm = max(float(np.linalg.norm(best_tangent)), 1e-6)
    return best_distance, best_point.astype(np.float32, copy=False), (best_tangent / tangent_norm).astype(np.float32, copy=False)


def _build_edge_attr(
    *,
    polyline_points_global: np.ndarray,
    object_centroid_global: np.ndarray,
    object_yaw_global: float,
    object_type: str,
    edge_type: str,
) -> np.ndarray:
    distance, nearest_point, tangent = _nearest_polyline_measure(polyline_points_global, object_centroid_global)
    polyline_len = float(np.linalg.norm(np.diff(polyline_points_global, axis=0), axis=-1).sum())
    cumulative = 0.0
    s_along = 0.0
    for seg_idx in range(polyline_points_global.shape[0] - 1):
        p0 = polyline_points_global[seg_idx]
        p1 = polyline_points_global[seg_idx + 1]
        segment = p1 - p0
        seg_len = float(np.linalg.norm(segment))
        if seg_len < 1e-6:
            continue
        denom = float(np.dot(segment, segment))
        alpha = float(np.dot(nearest_point - p0, segment) / max(denom, 1e-6))
        if 0.0 <= alpha <= 1.0:
            s_along = (cumulative + alpha * seg_len) / max(polyline_len, 1e-6)
            break
        cumulative += seg_len

    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    offset_vec = object_centroid_global - nearest_point
    lateral_offset = float(np.dot(offset_vec, normal))
    object_dir = np.asarray([np.cos(object_yaw_global), np.sin(object_yaw_global)], dtype=np.float32)
    cos_delta = float(np.dot(object_dir, tangent))
    sin_delta = float(object_dir[0] * tangent[1] - object_dir[1] * tangent[0])

    object_type_onehot = {
        "crosswalk": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "stop_line": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "traffic_light": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "carpark": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }[object_type]
    edge_type_onehot = {
        "associated": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "intersects": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "nearest": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    }[edge_type]

    return np.concatenate(
        [
            np.asarray(
                [
                    s_along,
                    lateral_offset,
                    distance,
                    cos_delta,
                    sin_delta,
                ],
                dtype=np.float32,
            ),
            object_type_onehot,
            edge_type_onehot,
        ],
        axis=0,
    )
