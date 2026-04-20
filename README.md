# Motion Prediction V1 на nuScenes

Репозиторий содержит автономную реализацию `Motion Prediction V1` без привязки к старому проектному архиву. Основная цель этой версии: оставить только актуальный код для подготовки данных, построения артефактов и обучения модели с универсальными путями через аргументы командной строки.

## Что внутри

- `motion_v1/` — основной V1-код: dataloader, модель, геометрия, категории
- `data/` — утилиты для raw/preprocessed датасетов и работы с `nuScenes`
- `preprocessing/` — offline-препроцессинг и построение артефактов
- `docs/PLAN.md` — план и зафиксированные решения по V1
- `train.py` — отдельная точка входа для обучения

## Структура

```text
motion_nuscenes/
├── motion_v1/
│   ├── __init__.py
│   ├── categories.py
│   ├── dataloader.py
│   ├── geometry.py
│   └── model.py
├── data/
│   ├── __init__.py
│   ├── motion_dataset.py
│   ├── nuscenes_utils.py
│   └── preprocessed_dataset.py
├── preprocessing/
│   ├── __init__.py
│   └── offline_preprocessing.py
├── docs/
│   └── PLAN.md
├── .gitignore
├── README.md
└── train.py
```

## Особенности реализации

- нет жёстких путей вида `/content/...` или локальных путей пользователя
- обучение вынесено в отдельный `train.py`
- `motion_v1` больше не импортирует код из исторического пакета
- визуализация в репозиторий не включена, чтобы структура оставалась чистой
- чекпоинты и артефакты по умолчанию игнорируются через `.gitignore`

## Быстрый старт

1. Установить зависимости `PyTorch`, `numpy`, `tqdm`, `nuscenes-devkit`, `shapely`.
2. Подготовить `train` и `val` артефакты `.pt`.
3. Запустить обучение из корня репозитория.

Пример:

```bash
python train.py --train-artifact artifacts/train.pt --val-artifact artifacts/val.pt
```

Чекпоинт по умолчанию будет сохранён в `artifacts/best_model.pt`.

## Замечания по данным

- датасет: `nuScenes`
- горизонт предсказания: `12` шагов
- карта и статические объекты учитываются на уровне сцены
- декодирование траекторий мультимодальное и anchor-based

Для raw-данных используются модули из `data/` и `preprocessing/`, а для обучения на готовых артефактах — `motion_v1/dataloader.py` и `train.py`.

## Лучший зафиксированный результат

Из текущих экспериментов сохранён следующий ориентир:

- лучший epoch: `14 / 30`
- `best_val_ade = 1.4193`
- `best_val_fde_l2 = 3.2450`
- `predicted_anchor_topk = 3`
- `soft_anchor_topk = 6`
- `cls_focal_gamma = 1.5`

Этот блок стоит воспринимать как справочную точку для V1, а не как окончательный benchmark.
