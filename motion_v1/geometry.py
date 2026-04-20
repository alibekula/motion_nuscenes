from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_to_rotation_matrix(yaw: float) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.asarray(
        [
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw],
        ],
        dtype=np.float32,
    )


def quaternion_yaw(rotation: list[float] | tuple[float, float, float, float]) -> float:
    w, x, y, z = (float(v) for v in rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def global_xy_to_ego(
    xy_global: np.ndarray,
    ego_xy_global: np.ndarray,
    ego_yaw_global: float,
) -> np.ndarray:
    rotation_global_from_ego = yaw_to_rotation_matrix(ego_yaw_global)
    return (rotation_global_from_ego.T @ (xy_global - ego_xy_global).astype(np.float32, copy=False)).astype(
        np.float32,
        copy=False,
    )


def interpolate_xy(a_xy: np.ndarray, b_xy: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * a_xy + alpha * b_xy).astype(np.float32, copy=False)


def interpolate_angle(a_yaw: float, b_yaw: float, alpha: float) -> float:
    a_vec = np.asarray([math.cos(a_yaw), math.sin(a_yaw)], dtype=np.float32)
    b_vec = np.asarray([math.cos(b_yaw), math.sin(b_yaw)], dtype=np.float32)
    vec = (1.0 - alpha) * a_vec + alpha * b_vec
    if float(np.linalg.norm(vec)) < 1e-6:
        return float(a_yaw)
    return float(math.atan2(vec[1], vec[0]))
