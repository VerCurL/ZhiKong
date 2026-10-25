"""Frozen neural baseline flight controller assets."""

__all__ = ["FrozenFlightController"]


def __getattr__(name):
    if name == "FrozenFlightController":
        from ..flight_controller import FrozenFlightController
        return FrozenFlightController
    raise AttributeError(name)
