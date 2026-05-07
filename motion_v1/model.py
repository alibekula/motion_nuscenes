from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _zero_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x.clamp_min(1e-6))))


def _yaw_to_rotation_ego_to_local(yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    return torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw], dim=-1),
            torch.stack([-sin_yaw, cos_yaw], dim=-1),
        ],
        dim=-2,
    )


@dataclass(frozen=True)
class V1ModelConfig:
    num_classes: int = 9
    history_feature_dim: int = 18
    hidden_dim: int = 192
    plan_dim: int = 64
    score_ff_mult: int = 4
    map_mlp_hidden_dim: int = 128
    gru_layers: int = 1
    tf_layers: int = 3
    tf_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    dt: float = 0.5
    k12: int = 64
    k6: int = 32
    r_max: float = 1.0
    d_min: float = 1e-3
    score_tau: float = 1.0
    cond_tau: float = 2.0
    cond_topk: int = 3
    polyline_point_dim: int = 4
    polyline_attr_dim: int = 8
    object_feature_dim: int = 10
    use_pose_encoding: bool = True
    use_relative_encoding: bool = True
    pose_radius_m: float = 80.0

    @property
    def motion_summary_dim(self) -> int:
        return 4


@dataclass(frozen=True)
class V1AnchorBank:
    full_bank: Any
    prefix_bank: Any
    r_max: float = 1.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "V1AnchorBank":
        return cls(
            full_bank=payload["full_bank"],
            prefix_bank=payload["prefix_bank"],
            r_max=float(payload.get("r_max", 1.0)),
        )


@dataclass(frozen=True)
class V1LossConfig:
    eps_low: float = 0.15
    eps_high: float = 0.5
    stage1_weight: float = 0.5
    stage2_weight: float = 1.0
    smooth_weight: float = 0.05
    cls_focal_gamma: float = 1.5
    stationary_cls_weight: float = 0.5
    soft_anchor_tau: float = 0.2
    soft_anchor_topk: int | None = 1


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )

    def forward(self, history_features: torch.Tensor, agent_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        if history_features.ndim != 4:
            raise ValueError("Expected history_features with shape [B, A, T, D].")

        batch_size, num_agents, _, _ = history_features.shape
        _, hidden = self.gru(history_features.reshape(batch_size * num_agents, history_features.shape[2], history_features.shape[3]))
        tokens = hidden[-1].reshape(batch_size, num_agents, self.hidden_dim)
        if agent_pad_mask is not None:
            tokens = tokens.masked_fill(agent_pad_mask.to(device=tokens.device, dtype=torch.bool).unsqueeze(-1), 0.0)
        return tokens


class MapEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        polyline_point_dim: int = 4,
        polyline_attr_dim: int = 8,
        object_feature_dim: int = 10,
        mlp_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.point_mlp = _mlp(polyline_point_dim, mlp_hidden_dim, hidden_dim)
        self.polyline_attr_mlp = _mlp(polyline_attr_dim, mlp_hidden_dim, hidden_dim)
        self.polyline_out = _mlp(hidden_dim * 2, mlp_hidden_dim, hidden_dim)
        self.object_mlp = _mlp(object_feature_dim, mlp_hidden_dim, hidden_dim)

    def encode_polylines(
        self,
        polyline_point_features: torch.Tensor,
        polyline_point_mask: torch.Tensor,
        polyline_attrs: torch.Tensor,
        polyline_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if polyline_point_features.ndim != 4:
            raise ValueError("Expected polyline_point_features with shape [B, P, N, D].")
        if polyline_point_mask.ndim != 3:
            raise ValueError("Expected polyline_point_mask with shape [B, P, N].")
        if polyline_attrs.ndim != 3:
            raise ValueError("Expected polyline_attrs with shape [B, P, D].")

        batch_size, num_polylines, num_points, _ = polyline_point_features.shape
        if num_polylines == 0 or num_points == 0:
            return polyline_point_features.new_zeros((batch_size, num_polylines, self.hidden_dim))

        point_tokens = self.point_mlp(polyline_point_features)
        point_mask = polyline_point_mask.to(device=polyline_point_features.device, dtype=torch.bool)
        pooled = point_tokens.masked_fill(~point_mask.unsqueeze(-1), float("-inf")).max(dim=2).values
        has_points = point_mask.any(dim=2, keepdim=True)
        pooled = torch.where(has_points, pooled, torch.zeros_like(pooled))

        attr_tokens = self.polyline_attr_mlp(polyline_attrs)
        poly_tokens = self.polyline_out(torch.cat([pooled, attr_tokens], dim=-1))
        if polyline_pad_mask is not None:
            poly_tokens = poly_tokens.masked_fill(
                polyline_pad_mask.to(device=poly_tokens.device, dtype=torch.bool).unsqueeze(-1),
                0.0,
            )
        return poly_tokens

    def encode_objects(
        self,
        object_features: torch.Tensor,
        object_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if object_features.ndim != 3:
            raise ValueError("Expected object_features with shape [B, O, D].")

        object_tokens = self.object_mlp(object_features)
        if object_pad_mask is not None:
            object_tokens = object_tokens.masked_fill(
                object_pad_mask.to(device=object_tokens.device, dtype=torch.bool).unsqueeze(-1),
                0.0,
            )
        return object_tokens

    def forward(
        self,
        polyline_point_features: torch.Tensor,
        polyline_point_mask: torch.Tensor,
        polyline_attrs: torch.Tensor,
        object_features: torch.Tensor,
        polyline_pad_mask: torch.Tensor | None = None,
        object_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        polyline_tokens = self.encode_polylines(
            polyline_point_features=polyline_point_features,
            polyline_point_mask=polyline_point_mask,
            polyline_attrs=polyline_attrs,
            polyline_pad_mask=polyline_pad_mask,
        )
        object_tokens = self.encode_objects(object_features=object_features, object_pad_mask=object_pad_mask)
        return polyline_tokens, object_tokens


class SceneEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.linear1 = nn.Linear(int(hidden_dim), int(hidden_dim) * int(ff_mult))
        self.linear2 = nn.Linear(int(hidden_dim) * int(ff_mult), int(hidden_dim))
        self.norm1 = nn.LayerNorm(int(hidden_dim))
        self.norm2 = nn.LayerNorm(int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))

    def forward(
        self,
        tokens: torch.Tensor,
        pad_mask: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_mask = None
        if attention_bias is not None:
            batch_size, num_heads, seq_len, _ = attention_bias.shape
            attn_mask = attention_bias.reshape(batch_size * num_heads, seq_len, seq_len)

        attended = self.self_attn(
            tokens,
            tokens,
            tokens,
            attn_mask=attn_mask,
            key_padding_mask=pad_mask,
            need_weights=False,
        )[0]
        tokens = self.norm1(tokens + self.dropout1(attended))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(tokens))))
        return self.norm2(tokens + self.dropout2(ff))


class SceneEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        use_pose_encoding: bool = True,
        use_relative_encoding: bool = True,
        pose_radius_m: float = 80.0,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.use_pose_encoding = bool(use_pose_encoding)
        self.use_relative_encoding = bool(use_relative_encoding)
        self.pose_radius_m = float(pose_radius_m)
        self.layers = nn.ModuleList(
            [
                SceneEncoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.type_embedding = nn.Embedding(3, int(hidden_dim))
        self.pose_embedding = _mlp(4, int(hidden_dim), int(hidden_dim))
        self.relative_bias_mlp = _mlp(7, int(hidden_dim), int(num_heads))
        _zero_last_linear(self.pose_embedding)
        _zero_last_linear(self.relative_bias_mlp)

    def forward(
        self,
        agent_tokens: torch.Tensor,
        agent_pad_mask: torch.Tensor,
        agent_poses: torch.Tensor | None = None,
        polyline_tokens: torch.Tensor | None = None,
        polyline_pad_mask: torch.Tensor | None = None,
        polyline_poses: torch.Tensor | None = None,
        object_tokens: torch.Tensor | None = None,
        object_pad_mask: torch.Tensor | None = None,
        object_poses: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if agent_tokens.ndim != 3:
            raise ValueError("Expected agent_tokens with shape [B, A, H].")
        if agent_pad_mask.ndim != 2:
            raise ValueError("Expected agent_pad_mask with shape [B, A].")

        batch_size, num_agents, hidden_dim = agent_tokens.shape
        polyline_tokens = self._default_optional_tokens(agent_tokens, polyline_tokens)
        object_tokens = self._default_optional_tokens(agent_tokens, object_tokens)
        polyline_pad_mask = self._default_optional_mask(batch_size, polyline_tokens, polyline_pad_mask, agent_tokens.device)
        object_pad_mask = self._default_optional_mask(batch_size, object_tokens, object_pad_mask, agent_tokens.device)
        agent_poses = self._default_optional_poses(batch_size, num_agents, agent_poses, agent_tokens.device, agent_tokens.dtype)
        polyline_poses = self._default_optional_poses(
            batch_size,
            polyline_tokens.shape[1],
            polyline_poses,
            agent_tokens.device,
            agent_tokens.dtype,
        )
        object_poses = self._default_optional_poses(
            batch_size,
            object_tokens.shape[1],
            object_poses,
            agent_tokens.device,
            agent_tokens.dtype,
        )
        agent_pad_mask = agent_pad_mask.to(device=agent_tokens.device, dtype=torch.bool)

        token_groups = []
        mask_groups = []
        type_groups = []
        pose_groups = []

        if num_agents > 0:
            token_groups.append(agent_tokens)
            mask_groups.append(agent_pad_mask)
            pose_groups.append(agent_poses)
            type_groups.append(
                self.type_embedding(torch.zeros((batch_size, num_agents), dtype=torch.long, device=agent_tokens.device))
            )
        if polyline_tokens.shape[1] > 0:
            token_groups.append(polyline_tokens)
            mask_groups.append(polyline_pad_mask)
            pose_groups.append(polyline_poses)
            type_groups.append(
                self.type_embedding(torch.ones((batch_size, polyline_tokens.shape[1]), dtype=torch.long, device=agent_tokens.device))
            )
        if object_tokens.shape[1] > 0:
            token_groups.append(object_tokens)
            mask_groups.append(object_pad_mask)
            pose_groups.append(object_poses)
            type_groups.append(
                self.type_embedding(torch.full((batch_size, object_tokens.shape[1]), 2, dtype=torch.long, device=agent_tokens.device))
            )

        if not token_groups:
            return agent_tokens.new_zeros((batch_size, num_agents, hidden_dim))

        poses = torch.cat(pose_groups, dim=1)
        tokens = torch.cat(token_groups, dim=1) + torch.cat(type_groups, dim=1)
        if self.use_pose_encoding:
            tokens = tokens + self._pose_embedding(poses)
        pad_mask = torch.cat(mask_groups, dim=1)

        safe_pad_mask = pad_mask.clone()
        fully_padded_rows = safe_pad_mask.all(dim=1)
        if torch.any(fully_padded_rows):
            safe_pad_mask[fully_padded_rows, 0] = False

        attention_bias = self._relative_attention_bias(poses) if self.use_relative_encoding else None
        encoded = tokens
        for layer in self.layers:
            encoded = layer(encoded, pad_mask=safe_pad_mask, attention_bias=attention_bias)
        encoded = self.norm(encoded)
        encoded = encoded.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return encoded[:, :num_agents]

    def _default_optional_tokens(self, reference_tokens: torch.Tensor, maybe_tokens: torch.Tensor | None) -> torch.Tensor:
        if maybe_tokens is not None:
            return maybe_tokens
        return reference_tokens.new_zeros((reference_tokens.shape[0], 0, reference_tokens.shape[-1]))

    def _default_optional_mask(
        self,
        batch_size: int,
        tokens: torch.Tensor,
        maybe_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if maybe_mask is not None:
            return maybe_mask.to(device=device, dtype=torch.bool)
        return torch.ones((batch_size, tokens.shape[1]), dtype=torch.bool, device=device)

    def _default_optional_poses(
        self,
        batch_size: int,
        num_tokens: int,
        maybe_poses: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if maybe_poses is not None:
            return maybe_poses.to(device=device, dtype=dtype)
        poses = torch.zeros((batch_size, num_tokens, 4), dtype=dtype, device=device)
        poses[..., 2] = 1.0
        return poses

    def _pose_embedding(self, poses: torch.Tensor) -> torch.Tensor:
        pose_features = poses.clone()
        pose_features[..., 0:2] = (pose_features[..., 0:2] / max(self.pose_radius_m, 1e-6)).clamp(-2.0, 2.0)
        return self.pose_embedding(pose_features)

    def _relative_attention_bias(self, poses: torch.Tensor) -> torch.Tensor:
        xy = poses[..., 0:2]
        cos_yaw = poses[..., 2]
        sin_yaw = poses[..., 3]
        delta = xy.unsqueeze(1) - xy.unsqueeze(2)
        scaled_delta = (delta / max(self.pose_radius_m, 1e-6)).clamp(-2.0, 2.0)
        distance = torch.linalg.norm(scaled_delta, dim=-1, keepdim=True).clamp(max=2.0)

        query_cos = cos_yaw.unsqueeze(2)
        query_sin = sin_yaw.unsqueeze(2)
        local_dx = scaled_delta[..., 0] * query_cos + scaled_delta[..., 1] * query_sin
        local_dy = -scaled_delta[..., 0] * query_sin + scaled_delta[..., 1] * query_cos
        relative_cos = cos_yaw.unsqueeze(1) * cos_yaw.unsqueeze(2) + sin_yaw.unsqueeze(1) * sin_yaw.unsqueeze(2)
        relative_sin = sin_yaw.unsqueeze(1) * cos_yaw.unsqueeze(2) - cos_yaw.unsqueeze(1) * sin_yaw.unsqueeze(2)

        features = torch.cat(
            [
                scaled_delta,
                torch.stack([local_dx, local_dy], dim=-1),
                distance,
                relative_cos.unsqueeze(-1),
                relative_sin.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.relative_bias_mlp(features).permute(0, 3, 1, 2).contiguous()


class AnchorDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        horizon: int,
        plan_dim: int,
        score_tau: float,
        score_ff_mult: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.score_tau = float(score_tau)
        score_hidden_dim = int(hidden_dim) * int(score_ff_mult)

        self.anchor_mlp = nn.Sequential(
            nn.Linear(self.horizon * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, score_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(score_hidden_dim, hidden_dim),
        )
        self.anchor_proj = nn.Sequential(
            nn.Linear(hidden_dim, score_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(score_hidden_dim, hidden_dim),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, score_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(score_hidden_dim, 1),
        )
        self.move_gate = nn.Sequential(
            nn.Linear(hidden_dim, score_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(score_hidden_dim, 1),
        )
        self.stat_score = nn.Sequential(
            nn.Linear(hidden_dim, score_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(score_hidden_dim, 1),
        )
        self.merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.progress_head = nn.Linear(hidden_dim, self.horizon)
        self.residual_head = nn.Linear(hidden_dim, self.horizon * 2)
        self.plan_head = nn.Linear(hidden_dim, plan_dim)
        self.stat_plan = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, plan_dim),
        )

    def forward(
        self,
        agent_tokens: torch.Tensor,
        anchors: torch.Tensor,
        v_last: torch.Tensor,
        d_min: float,
        r_max: float,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if anchors.ndim != 3 or anchors.shape[-1] != 2:
            raise ValueError("anchors must have shape [K, T, 2].")
        if anchors.shape[1] != self.horizon:
            raise ValueError(f"Expected anchors with horizon={self.horizon}, got {anchors.shape[1]}.")

        batch_size, num_agents, _ = agent_tokens.shape
        num_anchors = int(anchors.shape[0])
        anchor_tokens = self.anchor_mlp(anchors.reshape(num_anchors, self.horizon * 2))

        query = F.normalize(self.query_proj(agent_tokens), dim=-1)
        anchor_query = F.normalize(self.anchor_proj(anchor_tokens), dim=-1)
        dot_score = torch.einsum("bad,kd->bak", query, anchor_query) / max(self.score_tau, 1e-6)

        agent_expanded = agent_tokens.unsqueeze(2).expand(-1, -1, num_anchors, -1)
        anchor_expanded = anchor_tokens.unsqueeze(0).unsqueeze(0).expand(batch_size, num_agents, -1, -1)
        merged = self.merge(torch.cat([agent_expanded, anchor_expanded], dim=-1))
        pair_score = self.score_head(merged).squeeze(-1)
        move_logit = self.move_gate(agent_tokens)
        score_dir = dot_score + pair_score + move_logit
        score_stat = self.stat_score(agent_tokens) - move_logit
        score = torch.cat([score_dir, score_stat], dim=-1)

        raw_progress = self.progress_head(merged)
        raw_residual = self.residual_head(merged).reshape(batch_size, num_agents, num_anchors, self.horizon, 2)
        z_dir = self.plan_head(merged)
        z_stat = self.stat_plan(agent_tokens)

        d_init = (v_last * dt).clamp(min=d_min)
        progress = F.softplus(raw_progress + _inv_softplus(d_init[:, :, None, None]))
        residual = r_max * torch.tanh(raw_residual)

        tangent = anchors
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        base = torch.cumsum(progress.unsqueeze(-1) * tangent, dim=-2)
        traj = base + residual[..., 0:1] * tangent + residual[..., 1:2] * normal
        return score, traj, progress, z_dir, z_stat


class V1MotionModel(nn.Module):
    def __init__(self, cfg: V1ModelConfig, anchor_bank: V1AnchorBank | dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        if isinstance(anchor_bank, dict):
            anchor_bank = V1AnchorBank.from_payload(anchor_bank)

        self.register_buffer("A12", _to_tensor_anchor_bank(anchor_bank.full_bank))
        self.register_buffer("A6", _to_tensor_anchor_bank(anchor_bank.prefix_bank))
        self.r_max = float(anchor_bank.r_max if anchor_bank.r_max is not None else cfg.r_max)

        self.history_encoder = HistoryEncoder(
            input_dim=cfg.history_feature_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.gru_layers,
            dropout=cfg.dropout,
        )
        self.summary_proj = nn.Linear(cfg.hidden_dim + cfg.motion_summary_dim, cfg.hidden_dim)
        self.map_encoder = MapEncoder(
            hidden_dim=cfg.hidden_dim,
            polyline_point_dim=cfg.polyline_point_dim,
            polyline_attr_dim=cfg.polyline_attr_dim,
            object_feature_dim=cfg.object_feature_dim,
            mlp_hidden_dim=cfg.map_mlp_hidden_dim,
        )
        self.scene_encoder = SceneEncoder(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.tf_heads,
            num_layers=cfg.tf_layers,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
            use_pose_encoding=cfg.use_pose_encoding,
            use_relative_encoding=cfg.use_relative_encoding,
            pose_radius_m=cfg.pose_radius_m,
        )
        self.decoder1 = AnchorDecoder(
            cfg.hidden_dim,
            horizon=6,
            plan_dim=cfg.plan_dim,
            score_tau=cfg.score_tau,
            score_ff_mult=cfg.score_ff_mult,
            dropout=cfg.dropout,
        )
        self.decoder2 = AnchorDecoder(
            cfg.hidden_dim,
            horizon=12,
            plan_dim=cfg.plan_dim,
            score_tau=cfg.score_tau,
            score_ff_mult=cfg.score_ff_mult,
            dropout=cfg.dropout,
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim + cfg.plan_dim + 6, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        history_features = batch["history_features"]
        agent_pad_mask = batch["agent_pad_mask"].to(device=history_features.device, dtype=torch.bool)

        history_tokens = self.history_encoder(history_features, agent_pad_mask=agent_pad_mask)
        motion_summaries, v_last = self._motion_summaries(history_features)
        agent_tokens = self.summary_proj(torch.cat([history_tokens, motion_summaries], dim=-1))
        agent_tokens = agent_tokens.masked_fill(agent_pad_mask.unsqueeze(-1), 0.0)

        polyline_tokens, object_tokens = self.map_encoder(
            polyline_point_features=batch["polyline_point_features"],
            polyline_point_mask=batch["polyline_point_mask"],
            polyline_attrs=batch["polyline_attrs"],
            object_features=batch["object_features"],
            polyline_pad_mask=batch["polyline_pad_mask"],
            object_pad_mask=batch["object_pad_mask"],
        )
        agent_poses = self._agent_token_poses(history_features)
        polyline_poses = self._polyline_token_poses(batch["polyline_point_features"], batch["polyline_point_mask"])
        object_poses = self._object_token_poses(batch["object_features"])
        scene_agent_tokens = self.scene_encoder(
            agent_tokens=agent_tokens,
            agent_pad_mask=agent_pad_mask,
            agent_poses=agent_poses,
            polyline_tokens=polyline_tokens,
            polyline_pad_mask=batch["polyline_pad_mask"],
            polyline_poses=polyline_poses,
            object_tokens=object_tokens,
            object_pad_mask=batch["object_pad_mask"],
            object_poses=object_poses,
        )

        score6, traj6, ds6, z6_dir, z6_stat = self.decoder1(
            scene_agent_tokens,
            self.A6,
            v_last,
            self.cfg.d_min,
            self.r_max,
            self.cfg.dt,
        )

        conditioning = self._conditioning(
            score6=score6,
            traj6=traj6,
            z_dir=z6_dir,
            z_stat=z6_stat,
            v_last=v_last,
        )
        conditioned_agent_tokens = self.cond_mlp(torch.cat([scene_agent_tokens, conditioning], dim=-1))
        conditioned_agent_tokens = conditioned_agent_tokens.masked_fill(agent_pad_mask.unsqueeze(-1), 0.0)

        score12, traj12, ds12, z12_dir, z12_stat = self.decoder2(
            conditioned_agent_tokens,
            self.A12,
            v_last,
            self.cfg.d_min,
            self.r_max,
            self.cfg.dt,
        )

        stage1 = _build_stage_outputs(score6, traj6, ds6, self.A6)
        stage2 = _build_stage_outputs(score12, traj12, ds12, self.A12)
        return {
            "history_tokens": history_tokens,
            "motion_summaries": motion_summaries,
            "agent_tokens": agent_tokens,
            "polyline_tokens": polyline_tokens,
            "object_tokens": object_tokens,
            "agent_poses": agent_poses,
            "polyline_poses": polyline_poses,
            "object_poses": object_poses,
            "scene_agent_tokens": scene_agent_tokens,
            "conditioning_features": conditioning,
            "conditioned_agent_tokens": conditioned_agent_tokens,
            "z6_dir": z6_dir,
            "z6_stat": z6_stat,
            "z12_dir": z12_dir,
            "z12_stat": z12_stat,
            "v_last": v_last,
            "stage1": stage1,
            "stage2": stage2,
        }

    def _motion_summaries(self, history_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        velocity = history_features[..., 4:6]
        yaw_rate = history_features[..., 6]
        cos_yaw = history_features[..., 2]
        sin_yaw = history_features[..., 3]

        v_last = velocity[:, :, -1].norm(dim=-1)
        v_mean = velocity.norm(dim=-1).mean(dim=-1)
        u = velocity[..., 0] * cos_yaw + velocity[..., 1] * sin_yaw
        a_long = (u[:, :, -1] - u[:, :, 0]) / max((history_features.shape[2] - 1) * self.cfg.dt, 1e-6)
        yaw_rate_mean = yaw_rate.mean(dim=-1)

        summaries = torch.stack([v_last, v_mean, a_long, yaw_rate_mean], dim=-1)
        return summaries, v_last

    def _agent_token_poses(self, history_features: torch.Tensor) -> torch.Tensor:
        last = history_features[:, :, -1]
        return torch.cat([last[..., 0:2], last[..., 2:4]], dim=-1)

    def _polyline_token_poses(
        self,
        polyline_point_features: torch.Tensor,
        polyline_point_mask: torch.Tensor,
    ) -> torch.Tensor:
        point_mask = polyline_point_mask.to(device=polyline_point_features.device, dtype=polyline_point_features.dtype)
        denom = point_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        xy = (polyline_point_features[..., 0:2] * point_mask.unsqueeze(-1)).sum(dim=2) / denom
        tangent = (polyline_point_features[..., 2:4] * point_mask.unsqueeze(-1)).sum(dim=2)
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        default_dir = torch.zeros_like(tangent)
        default_dir[..., 0] = 1.0
        direction = torch.where(tangent_norm > 1e-6, tangent / tangent_norm.clamp_min(1e-6), default_dir)
        return torch.cat([xy, direction], dim=-1)

    def _object_token_poses(self, object_features: torch.Tensor) -> torch.Tensor:
        if object_features.shape[1] == 0:
            return object_features.new_zeros((object_features.shape[0], 0, 4))
        return object_features[..., 0:4]

    def _conditioning(
        self,
        score6: torch.Tensor,
        traj6: torch.Tensor,
        z_dir: torch.Tensor,
        z_stat: torch.Tensor,
        v_last: torch.Tensor,
    ) -> torch.Tensor:
        num_classes = int(score6.shape[-1])
        topk = min(int(self.cfg.cond_topk), num_classes)

        pi = F.softmax(score6 / max(self.cfg.cond_tau, 1e-6), dim=-1)
        if topk < num_classes:
            _, topk_idx = pi.topk(topk, dim=-1)
            topk_mask = torch.zeros_like(pi).scatter_(-1, topk_idx, 1.0)
            pi = pi * topk_mask
        pi = (pi / pi.sum(-1, keepdim=True).clamp(min=1e-8)).detach()

        z_all = torch.cat([z_dir, z_stat.unsqueeze(2)], dim=2)
        zero_traj = traj6.new_zeros(traj6.shape[0], traj6.shape[1], 1, traj6.shape[3], 2)
        traj_all = torch.cat([traj6, zero_traj], dim=2)

        expected_plan = (pi.unsqueeze(-1) * z_all).sum(dim=2)
        end_pos = (pi.unsqueeze(-1) * traj_all[:, :, :, 5]).sum(dim=2)
        end_delta = (pi.unsqueeze(-1) * (traj_all[:, :, :, 5] - traj_all[:, :, :, 4])).sum(dim=2)
        end_dir = end_delta / end_delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        end_speed = end_delta.norm(dim=-1) / max(self.cfg.dt, 1e-6)
        avg_acc = (end_speed - v_last) / 3.0

        return torch.cat([expected_plan, end_pos, end_dir, end_speed.unsqueeze(-1), avg_acc.unsqueeze(-1)], dim=-1)


def _to_tensor_anchor_bank(bank: Any) -> torch.Tensor:
    tensor = bank if isinstance(bank, torch.Tensor) else torch.as_tensor(bank)
    tensor = tensor.to(dtype=torch.float32)
    if tensor.ndim != 3 or tensor.shape[-1] != 2:
        raise ValueError("Anchor bank must have shape [K, T, 2].")
    return tensor


def _build_stage_outputs(
    logits: torch.Tensor,
    directional_local_trajectories: torch.Tensor,
    directional_progress: torch.Tensor,
    directional_anchor_bank: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero_traj = directional_local_trajectories.new_zeros(
        directional_local_trajectories.shape[0],
        directional_local_trajectories.shape[1],
        1,
        directional_local_trajectories.shape[3],
        directional_local_trajectories.shape[4],
    )
    all_local = torch.cat([directional_local_trajectories, zero_traj], dim=2)
    best_class_idx = logits.argmax(dim=-1)
    best_local_trajectory = _gather_mode(all_local, best_class_idx)
    return {
        "logits": logits,
        "directional_anchor_bank": directional_anchor_bank,
        "directional_local_trajectories": directional_local_trajectories,
        "directional_progress": directional_progress,
        "best_class_idx": best_class_idx,
        "best_local_trajectory": best_local_trajectory,
    }


def future_positions_local_from_history(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    history = batch["history_features"]
    future_ego = batch["future_positions_ego"]
    last_xy = history[:, :, -1, 0:2]
    last_yaw = torch.atan2(history[:, :, -1, 3], history[:, :, -1, 2])
    rotation = _yaw_to_rotation_ego_to_local(last_yaw)
    rel_future = future_ego - last_xy.unsqueeze(2)
    return torch.matmul(rotation.unsqueeze(2), rel_future.unsqueeze(-1)).squeeze(-1)


def compute_stationary_direction_weight(
    gt_future_local: torch.Tensor,
    horizon: int,
    eps_low: float,
    eps_high: float,
) -> torch.Tensor:
    if horizon <= 0 or horizon > gt_future_local.shape[-2]:
        raise ValueError("horizon must be within gt_future_local length.")
    if eps_high <= eps_low:
        raise ValueError("eps_high must be greater than eps_low.")

    displacement = torch.linalg.norm(gt_future_local[..., horizon - 1, :], dim=-1)
    return ((displacement - eps_low) / (eps_high - eps_low)).clamp(0.0, 1.0)


def compute_anchor_direction_distance(
    gt_future_local: torch.Tensor,
    anchor_bank: torch.Tensor,
) -> torch.Tensor:
    if gt_future_local.ndim != 4:
        raise ValueError("Expected gt_future_local with shape [B, A, H, 2].")
    if anchor_bank.ndim != 3 or anchor_bank.shape[-1] != 2:
        raise ValueError("Expected anchor_bank with shape [K, H, 2].")

    horizon = int(anchor_bank.shape[1])
    gt = gt_future_local[..., :horizon, :]
    prev = torch.cat([torch.zeros_like(gt[..., :1, :]), gt[..., :-1, :]], dim=-2)
    delta = gt - prev
    step_len = torch.linalg.norm(delta, dim=-1)
    step_weight = step_len / step_len.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    step_dir = delta / step_len.unsqueeze(-1).clamp_min(1e-6)

    bank = anchor_bank.to(device=gt.device, dtype=gt.dtype).view(1, 1, *anchor_bank.shape)
    dot = (step_dir.unsqueeze(2) * bank).sum(dim=-1)
    return (step_weight.unsqueeze(2) * (1.0 - dot)).sum(dim=-1)


def build_soft_anchor_targets(
    anchor_distance: torch.Tensor,
    direction_weight: torch.Tensor,
    tau: float,
    topk: int | None = None,
) -> torch.Tensor:
    if anchor_distance.ndim != direction_weight.ndim + 1:
        raise ValueError("anchor_distance must have shape [B, A, K] when direction_weight has shape [B, A].")
    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    num_anchors = int(anchor_distance.shape[-1])
    directional_logits = -anchor_distance / float(tau)
    if topk is not None and topk > 0 and topk < num_anchors:
        topk_idx = directional_logits.topk(topk, dim=-1).indices
        keep = torch.zeros_like(directional_logits).scatter_(-1, topk_idx, 1.0)
        directional_logits = directional_logits.masked_fill(keep == 0, float("-inf"))

    directional = F.softmax(directional_logits, dim=-1).to(dtype=direction_weight.dtype)
    stationary = 1.0 - direction_weight
    target = torch.zeros(
        (*direction_weight.shape, num_anchors + 1),
        dtype=direction_weight.dtype,
        device=direction_weight.device,
    )
    target[..., :num_anchors] = directional * direction_weight.unsqueeze(-1)
    target[..., -1] = stationary
    return target


def soft_target_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
    gamma: float = 0.0,
) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("logits and target must have the same shape.")

    log_prob = F.log_softmax(logits, dim=-1)
    prob = log_prob.exp()
    loss_per_class = -(target * log_prob)

    if gamma > 0.0:
        focal = (1.0 - prob).clamp_min(1e-6).pow(float(gamma))
        loss_per_class = loss_per_class * focal

    if class_weight is not None:
        weight = class_weight.view(*([1] * (loss_per_class.ndim - 1)), -1)
        loss_per_class = loss_per_class * weight

    loss = loss_per_class.sum(dim=-1)
    if mask is None:
        return loss.mean()

    mask = mask.to(dtype=torch.bool)
    if not torch.any(mask):
        return logits.sum() * 0.0
    return loss[mask].mean()


def compute_v1_losses(
    outputs: dict[str, Any],
    batch: dict[str, torch.Tensor],
    cfg: V1LossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = cfg or V1LossConfig()
    if "stage1" not in outputs or "stage2" not in outputs:
        raise ValueError("Expected outputs to contain 'stage1' and 'stage2'.")

    gt_future_local = future_positions_local_from_history(batch)
    valid_agent_mask = ~batch["agent_pad_mask"].to(dtype=torch.bool)

    stage1_terms = _compute_stage_loss_terms(
        stage_outputs=outputs["stage1"],
        gt_future_local=gt_future_local,
        valid_agent_mask=valid_agent_mask,
        eps_low=cfg.eps_low,
        eps_high=cfg.eps_high,
        cls_focal_gamma=cfg.cls_focal_gamma,
        stationary_cls_weight=cfg.stationary_cls_weight,
        soft_anchor_tau=cfg.soft_anchor_tau,
        soft_anchor_topk=cfg.soft_anchor_topk,
    )
    stage2_terms = _compute_stage_loss_terms(
        stage_outputs=outputs["stage2"],
        gt_future_local=gt_future_local,
        valid_agent_mask=valid_agent_mask,
        eps_low=cfg.eps_low,
        eps_high=cfg.eps_high,
        cls_focal_gamma=cfg.cls_focal_gamma,
        stationary_cls_weight=cfg.stationary_cls_weight,
        soft_anchor_tau=cfg.soft_anchor_tau,
        soft_anchor_topk=cfg.soft_anchor_topk,
    )

    total_loss = (
        cfg.stage1_weight
        * (stage1_terms["cls"] + stage1_terms["pos"] + stage1_terms["fde"] + cfg.smooth_weight * stage1_terms["smooth"])
        + cfg.stage2_weight
        * (stage2_terms["cls"] + stage2_terms["pos"] + stage2_terms["fde"] + cfg.smooth_weight * stage2_terms["smooth"])
    )

    stage1_metrics = _compute_prediction_metrics(outputs["stage1"]["best_local_trajectory"], gt_future_local, valid_agent_mask)
    stage2_metrics = _compute_prediction_metrics(outputs["stage2"]["best_local_trajectory"], gt_future_local, valid_agent_mask)
    metrics = {
        "stage1_cls": float(stage1_terms["cls"].item()),
        "stage1_pos": float(stage1_terms["pos"].item()),
        "stage1_fde": float(stage1_terms["fde"].item()),
        "stage1_smooth": float(stage1_terms["smooth"].item()),
        "stage1_top1_ade": stage1_metrics["top1_ade"],
        "stage1_top1_fde_l2": stage1_metrics["top1_fde_l2"],
        "stage2_cls": float(stage2_terms["cls"].item()),
        "stage2_pos": float(stage2_terms["pos"].item()),
        "stage2_fde": float(stage2_terms["fde"].item()),
        "stage2_smooth": float(stage2_terms["smooth"].item()),
        "stage2_top1_ade": stage2_metrics["top1_ade"],
        "stage2_top1_fde_l2": stage2_metrics["top1_fde_l2"],
        "total": float(total_loss.item()),
    }
    return total_loss, metrics


def _compute_stage_loss_terms(
    stage_outputs: dict[str, torch.Tensor],
    gt_future_local: torch.Tensor,
    valid_agent_mask: torch.Tensor,
    eps_low: float,
    eps_high: float,
    cls_focal_gamma: float,
    stationary_cls_weight: float,
    soft_anchor_tau: float,
    soft_anchor_topk: int | None,
) -> dict[str, torch.Tensor]:
    horizon = int(stage_outputs["directional_anchor_bank"].shape[1])
    gt_local = gt_future_local[..., :horizon, :]
    direction_weight = compute_stationary_direction_weight(
        gt_future_local=gt_future_local,
        horizon=horizon,
        eps_low=eps_low,
        eps_high=eps_high,
    )
    anchor_distance = compute_anchor_direction_distance(
        gt_future_local=gt_future_local,
        anchor_bank=stage_outputs["directional_anchor_bank"],
    )
    cls_target = build_soft_anchor_targets(
        anchor_distance=anchor_distance,
        direction_weight=direction_weight,
        tau=soft_anchor_tau,
        topk=soft_anchor_topk,
    )

    num_classes = int(stage_outputs["logits"].shape[-1])
    class_weight = stage_outputs["logits"].new_ones((num_classes,))
    class_weight[-1] = float(stationary_cls_weight)
    class_weight = class_weight * (num_classes / class_weight.sum().clamp_min(1e-6))

    cls_loss = soft_target_cross_entropy(
        stage_outputs["logits"],
        cls_target,
        mask=valid_agent_mask,
        class_weight=class_weight,
        gamma=cls_focal_gamma,
    )
    gathered_local, gathered_progress = _select_regression_prediction(
        stage_outputs=stage_outputs,
    )

    pos_loss = _weighted_regression_loss(
        prediction=gathered_local,
        target=gt_local,
        weight=direction_weight,
        valid_agent_mask=valid_agent_mask,
    )
    fde_loss = _weighted_regression_loss(
        prediction=gathered_local[..., -1, :],
        target=gt_local[..., -1, :],
        weight=direction_weight,
        valid_agent_mask=valid_agent_mask,
    )
    if gathered_progress.shape[-1] > 1:
        smooth_delta = gathered_progress[..., 1:] - gathered_progress[..., :-1]
        smooth_loss = _weighted_regression_loss(
            prediction=smooth_delta,
            target=torch.zeros_like(smooth_delta),
            weight=direction_weight,
            valid_agent_mask=valid_agent_mask,
        )
    else:
        smooth_loss = cls_loss * 0.0

    return {
        "cls": cls_loss,
        "pos": pos_loss,
        "fde": fde_loss,
        "smooth": smooth_loss,
    }


def _select_regression_prediction(
    stage_outputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    directional_local = stage_outputs["directional_local_trajectories"]
    directional_progress = stage_outputs["directional_progress"]
    anchor_idx = stage_outputs["logits"][..., :-1].argmax(dim=-1)
    return _gather_mode(directional_local, anchor_idx), _gather_mode(directional_progress, anchor_idx)


def _gather_mode(values: torch.Tensor, mode_idx: torch.Tensor) -> torch.Tensor:
    if values.ndim < 4:
        raise ValueError("Expected values with shape [B, A, K, ...].")
    if mode_idx.shape != values.shape[:2]:
        raise ValueError("mode_idx must have shape [B, A].")

    tail_shape = values.shape[3:]
    index = mode_idx.long().view(*mode_idx.shape, 1, *([1] * len(tail_shape)))
    index = index.expand(*mode_idx.shape, 1, *tail_shape)
    return torch.gather(values, dim=2, index=index).squeeze(2)


def _weighted_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    valid_agent_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if weight.shape != valid_agent_mask.shape:
        raise ValueError("weight and valid_agent_mask must have shape [B, A].")

    per_item = F.smooth_l1_loss(prediction, target, reduction="none")
    if prediction.ndim > weight.ndim and prediction.shape[-1] == 2:
        per_item = per_item.sum(dim=-1)
    if per_item.ndim > weight.ndim:
        reduce_dims = tuple(range(weight.ndim, per_item.ndim))
        per_item = per_item.mean(dim=reduce_dims)

    valid_weight = weight * valid_agent_mask.to(dtype=weight.dtype)
    normalizer = valid_weight.sum().clamp_min(1e-6)
    return (per_item * valid_weight).sum() / normalizer


def _compute_prediction_metrics(
    prediction: torch.Tensor,
    gt_future_local: torch.Tensor,
    valid_agent_mask: torch.Tensor,
) -> dict[str, float]:
    horizon = prediction.shape[-2]
    target = gt_future_local[..., :horizon, :]
    if not torch.any(valid_agent_mask):
        return {"top1_ade": 0.0, "top1_fde_l2": 0.0}

    l2 = torch.linalg.norm(prediction - target, dim=-1)
    ade = float(l2[valid_agent_mask].mean().item())
    fde = float(torch.linalg.norm(prediction[..., -1, :] - target[..., -1, :], dim=-1)[valid_agent_mask].mean().item())
    return {"top1_ade": ade, "top1_fde_l2": fde}
