# motion_prediction_v1

Сценовый baseline для multi-agent motion prediction на `nuScenes`.

## Overview

Этот проект собран как чистый и воспроизводимый baseline для предсказания траекторий в сценах автономного вождения. Базовый принцип здесь простой: один sample соответствует одному keyframe, все агенты и элементы карты приводятся к текущей ego-системе координат, а модель работает с полностью подготовленными scene-level тензорами без online-преобразований в training loop.

Модель предсказывает `K=64` мультимодальных будущих траекторий для каждого агента на горизонте `6` секунд через двухэтапный anchor-based decoder. В финальном артефакте используется окно `history=4` / `future=12` кадров при `dt=0.5`.

## Results

| Metric | Value |
|--------|-------|
| val ADE | 1.4193 |
| val FDE | 3.2450 |

Лучший результат относится к финальной V1-конфигурации с offline-артефактами и anchor-based routing.

## Architecture

```text
HistoryEncoder (GRU)
    └── 18-мерный признак на шаг: x, y, cos/sin yaw, vx, vy, yaw_rate, width, length, class_onehot[9]
    └── явная кинематическая сводка: v_last, mean_speed, accel, mean_yaw_rate

MapEncoder
    └── polylines: point MLP + attribute MLP (lane, lane_connector, divider)
    └── objects: MLP (ped_crossing, stop_line, traffic_light, carpark_area)

SceneEncoder (TransformerEncoder)
    └── agent tokens + polyline tokens + object tokens в общей ego-системе координат

AnchorDecoder (two-stage)
    └── stage1: 3-секундный prefix → компактный plan vector (A6 bank)
    └── stage2: полный 6-секундный rollout с условием от stage1 (A12 bank)
```

`A6` и `A12` относятся к числу future-кадров в anchor bank: 6 кадров для 3-секундного prefix и 12 кадров для полного 6-секундного прогноза. Это не размер history window.

## Data Pipeline

Offline preprocessing отделяет подготовку данных от обучения:

1. `Scene windows`
   Для финального лучшего запуска использовались окна фиксированной длины: `history=4` кадра и `future=12` кадров при `dt=0.5`.
2. `Agent filtering`
   В артефакты попадают только агенты с полным треком на всём нужном интервале, без partial trajectories и без runtime-достройки пропусков.
3. `Map store`
   Строится один раз на уровень `map_name`; lane/connector/divider геометрия векторизуется заранее, а локальный контекст сцены выбирается вокруг ego-позиции.
4. `Anchor bank`
   Строится offline по agent-local directional profiles и используется как routing-пространство для мультимодального предсказания.
5. `Artifacts`
   Сохраняются на диск, после чего train loader в основном только читает, паддит и возвращает готовые тензоры.

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

| Setting | val ADE |
|---------|---------|
| hard WTA | 1.4378 |
| predicted_topk=6 | 1.4262 |
| predicted_topk=3 | 1.4193 |

## Qualitative Example

Ниже показан пример BEV-сцены с `history(4)`, `gt future(12)` и предсказанной траекторией в ego-frame.

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

`motion_v1` и `data` здесь намеренно разделены. `motion_v1` содержит модель, V1-загрузчик и общие геометрические утилиты, а `data` отвечает за raw/preprocessed dataset abstractions и `nuScenes`-специфичные вспомогательные функции, которые используются офлайн-препроцессингом и частью пайплайна подготовки данных.

```text
motion_nuscenes/
├── motion_v1/               # модель, V1 dataloader и базовые геометрические утилиты
│   ├── dataloader.py
│   ├── model.py
│   ├── categories.py
│   ├── geometry.py
│   └── __init__.py
├── data/                    # raw/preprocessed dataset wrappers и nuScenes utilities
│   ├── motion_dataset.py
│   ├── preprocessed_dataset.py
│   ├── nuscenes_utils.py
│   └── __init__.py
├── preprocessing/           # офлайн-генерация артефактов и anchor bank
│   ├── offline_preprocessing.py
│   └── __init__.py
├── docs/
│   └── assets/
│       └── bev_example.png
├── .gitignore
├── README.md
└── train.py
```
