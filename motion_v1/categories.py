from __future__ import annotations


AGENT_CLASS_NAMES = (
    "ego",
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
)

EGO_CLASS_NAME = "ego"

_PEDESTRIAN_RAW_NAMES = {
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.police_officer",
}


def normalize_motion_agent_class(name: str | None) -> str | None:
    if not name:
        return None

    value = str(name).lower()

    if value in {
        "vehicle.car",
        "vehicle.emergency.ambulance",
        "vehicle.emergency.police",
    }:
        return "car"
    if value == "vehicle.truck":
        return "truck"
    if value in {"vehicle.bus.bendy", "vehicle.bus.rigid"}:
        return "bus"
    if value == "vehicle.trailer":
        return "trailer"
    if value == "vehicle.construction":
        return "construction_vehicle"
    if value in _PEDESTRIAN_RAW_NAMES:
        return "pedestrian"
    if value == "vehicle.motorcycle":
        return "motorcycle"
    if value == "vehicle.bicycle":
        return "bicycle"
    return None
