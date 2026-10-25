"""Flight-controller registry for the high-level policy adapter."""

FLIGHT_CONTROLLER_NAMES = ("baseline", "pid")


def validate_flight_controller(name):
    if name not in FLIGHT_CONTROLLER_NAMES:
        raise ValueError(f"Unknown flight controller {name!r}; choose {FLIGHT_CONTROLLER_NAMES}")
    return name
