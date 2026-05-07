# motion_prediction_v1

Сценовый baseline для multi-agent motion prediction на `nuScenes`.

## Overview

Этот проект собран как чистый и воспроизводимый baseline для предсказания траекторий в сценах автономного вождения. Базовый принцип здесь простой: один sample соответствует одному V1 scene window с reference keyframe, все агенты и элементы карты приводятся к ego-системе координат этого reference frame, а модель работает с полностью подготовленными scene-level тензорами без runtime-достройки пропусков в training loop.

Модель предсказывает `K=64` мультимодальных будущих траекторий для каждого агента на горизонте `6` секунд через двухэтапный anchor-based decoder.

## Results

| Setting | val top1 ADE | val top1 FDE |
|---------|--------------|--------------|
| baseline, dropout=0.15, no augmentation | 1.3364 | 3.0463 |
| dropout=0.10, rotation/translation aug + yaw noise | 1.3119 | 2.9974 |
| dropout=0.00, rotation/translation aug + yaw noise | 1.3212 | 3.0239 |
| dropout=0.20, rotation/translation aug + yaw noise | 1.3504 | 3.1003 |

Лучший результат относится к финальной V1-конфигурации с artifact payload и anchor-based routing.

Leaderboard-like validation metrics for `best_model_noise_do01.pt`, computed with `evaluate_leaderboard.py` on `val_v1_artifact.pt`, excluding ego agents and using a 2m miss threshold:

| Metric | Value |
|--------|-------|
| Top1ADE | 1.0748 |
| MinADE_5 | 0.9396 |
| MinADE_10 | 0.8764 |
| MinFDE_1 | 2.4265 |
| MissRateTopK_2_5 | 0.2055 |
| MissRateTopK_2_10 | 0.1890 |
| num_eval_agents | 26894 |

These are not official nuScenes leaderboard scores: the evaluation uses the local artifact/filtering pipeline. Official submission still requires predictions for the `get_prediction_challenge_split("val")` agent list in the nuScenes `Prediction` JSON format.

## Data Pipeline

Artifact build отделяет подготовку данных от обучения. В репозитории поддерживается один контракт данных: V1 artifact payload из `motion_v1/dataloader.py`.

```mermaid
flowchart TD
    A["nuScenes raw scenes and maps"] --> B["build_v1_map_store"]
    B --> C["global map store by map_name"]

    A --> D["V1WindowDataset"]
    D --> E["_index_scenes"]
    E --> F["ego frames by scene"]
    E --> G["agent frames by scene"]
    D --> H["_build_windows"]
    H --> I["fixed scene windows"]

    C --> J["_build_sample(idx)"]
    F --> J
    G --> J
    I --> J

    J --> K["_collect_agents"]
    K --> K2["_build_agent_entry"]
    K2 --> L["agent tensors"]

    J --> M["_select_local_map"]
    M --> N["map tensors"]

    L --> P["V1 sample dict"]
    N --> P
    P --> Q["build_artifact_payload"]
    Q --> R["V1 artifact payload"]

    R --> S["V1ArtifactDataset.__getitem__ + optional augmentation"]
    S --> T["collate_v1_batch"]
    T --> U["padded scene batch"]
```

1. `Scene windows`
   Для финального лучшего запуска использовались окна с историей и будущим фиксированной длины.
2. `Agent selection`
   Целевые агенты выбираются из reference keyframe; для каждого собирается фиксированное history+future окно.
3. `Map store`
   Строится один раз на уровень `map_name`; lane/connector/divider геометрия векторизуется заранее, а локальный контекст сцены выбирается вокруг ego-позиции.
4. `Anchor bank`
   Строится k-means++/k-means по agent-local directional profiles и используется как routing-пространство для мультимодального предсказания.
5. `Artifacts`
   Сохраняются на диск, после чего train loader в основном только читает, паддит и возвращает готовые тензоры.

## Architecture

```mermaid
flowchart TD
    A["padded scene batch"] --> B["HistoryEncoder"]
    A --> C["motion summaries"]
    A --> D["MapEncoder"]
    A --> Q["token poses"]

    B --> E["history tokens"]
    C --> F["agent tokens"]
    E --> F

    D --> G["polyline tokens"]
    D --> H["object tokens"]

    F --> I["SceneEncoder with pose embeddings and relative bias"]
    G --> I
    H --> I
    Q --> I

    I --> J["scene agent tokens"]
    J --> K["AnchorDecoder stage1 / A6"]
    K --> L["6-step scores and trajectories"]

    L --> M["conditioning"]
    J --> M
    M --> N["conditioned agent tokens"]

    N --> O["AnchorDecoder stage2 / A12"]
    O --> P["12-step scores and trajectories"]
```

## Loss Configuration

| Parameter | Value |
|-----------|-------|
| soft_anchor_topk | 6 |
| soft_anchor_tau | 0.2 |
| predicted_topk | 3 |
| predicted_anchor_detach | True |
| cls_focal_gamma | 1.5 |
| stationary_cls_weight | 0.4 |
| gt_cond_weight | 0 |

В финальной схеме stage2 обучается на предсказанном выходе stage1, а `detach=True` на mixture-routing помогает не портить классификационную ветку регрессионными градиентами.

## Key Ablations

| Setting | val top1 ADE |
|---------|---------|
| baseline, dropout=0.15, no augmentation | 1.3364 |
| train aug prob=0.2, rotation only | 1.3442 |
| stage1_weight=0.25 | 1.3492 |
| dropout=0.20, no augmentation | 1.3621 |
| dropout=0.10, rotation/translation aug + yaw noise | 1.3119 |
| dropout=0.00, rotation/translation aug + yaw noise | 1.3212 |

## Qualitative Example

Ниже показан пример BEV-сцены с историей движения, ground-truth будущим и предсказанной траекторией в ego-frame.

![BEV trajectory example](docs/assets/bev_example.png)

## Requirements

```text
torch
nuscenes-devkit
numpy
tqdm
shapely
```

## Project Structure

`evaluate_leaderboard.py` computes local leaderboard-like validation metrics from a saved checkpoint and artifact.

`motion_v1` содержит единственный актуальный пайплайн: сборку V1 artifact payload, dataloader, модель, loss и геометрические утилиты. `data` оставлен только для небольших `nuScenes`-специфичных helper-функций, которые нужны V1-загрузчику.

```text
motion_nuscenes/
├── motion_v1/               # модель, V1 dataloader и базовые геометрические утилиты
│   ├── dataloader.py
│   ├── model.py
│   ├── categories.py
│   ├── geometry.py
│   └── __init__.py
├── data/                    # nuScenes utilities для V1 dataloader
│   ├── nuscenes_utils.py
│   └── __init__.py
├── docs/
│   └── assets/
│       └── bev_example.png
├── .gitignore
├── README.md
└── train.py
```
