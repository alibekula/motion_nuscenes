from __future__ import annotations

from typing import Iterable

from nuscenes.nuscenes import NuScenes


def init_nuscenes(
    dataroot: str = "./nuscenes",
    version: str = "v1.0-trainval",
    verbose: bool = False,
) -> NuScenes:
    return NuScenes(version=version, dataroot=dataroot, verbose=verbose)


def select_split_tokens(
    nusc: NuScenes,
    split_name: str,
    scene_limit: int | None = None,
) -> list[str]:
    from nuscenes.utils.splits import create_splits_scenes

    split_scene_names = set(create_splits_scenes()[split_name])
    scenes = [scene for scene in nusc.scene if scene["name"] in split_scene_names]
    if scene_limit is not None:
        scenes = scenes[: int(scene_limit)]

    sample_tokens: list[str] = []
    for scene in scenes:
        token = scene["first_sample_token"]
        while token:
            sample_tokens.append(token)
            token = nusc.get("sample", token)["next"]
    return sample_tokens


def build_scene_timelines(nusc: NuScenes) -> tuple[dict[str, list[str]], dict[str, int], dict[str, str]]:
    scene_to_tokens: dict[str, list[str]] = {}
    sample_to_index: dict[str, int] = {}
    sample_to_scene: dict[str, str] = {}

    for scene in nusc.scene:
        scene_token = str(scene["token"])
        tokens: list[str] = []
        token = scene["first_sample_token"]
        while token:
            sample_to_index[token] = len(tokens)
            sample_to_scene[token] = scene_token
            tokens.append(token)
            token = nusc.get("sample", token)["next"]
        scene_to_tokens[scene_token] = tokens

    return scene_to_tokens, sample_to_index, sample_to_scene


def get_scene_sample_tokens(nusc: NuScenes, scenes: Iterable[dict]) -> list[str]:
    sample_tokens: list[str] = []
    for scene in scenes:
        token = scene["first_sample_token"]
        while token:
            sample_tokens.append(token)
            token = nusc.get("sample", token)["next"]
    return sample_tokens
