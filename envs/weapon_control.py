"""Shared legal weapon requests and platform commands for masks and adapters."""

from algorithms.utils.action_spec import WEAPON_CHOICES
from .observation_encoder import alive, digits_mask, offset
from .utils.utils import decode_aircraft_id


def _ready_slots(plane, prefix, capacity, ammo):
    fields = [f"{prefix}{index+1}_CanReload" for index in range(capacity)]
    if any(field in plane for field in fields):
        return [index for index, field in enumerate(fields) if int(plane.get(field, 0)) == 1]
    # Legacy telemetry can omit rail flags; infer the next sequential rail.
    return [min(capacity-1, max(0, capacity-ammo))] if ammo > 0 else []


def _locked_indices(value, slots):
    try:
        value = int(value)
    except (ValueError, TypeError, OverflowError):
        return []
    if value < 0:
        return []
    return [int(digit) for digit in str(value).zfill(slots)[::-1][:slots]]


def _amraam_targets(state, own_name):
    """Only this aircraft's currently flying radar missiles reserve targets."""
    targets = set()
    for identifier, missile in state.get("missile", {}).items():
        if not str(identifier).upper().startswith("AMRAAM_") or not isinstance(missile, dict):
            continue
        if int(missile.get("Status", -1)) == 0 and decode_aircraft_id(missile.get("Owner")) == own_name:
            target = decode_aircraft_id(missile.get("Target"))
            if target is not None:
                targets.add(target)
    return targets


def weapon_plans(state, side, own_name, encoder, profile, target_name=None):
    """Pure plans: all launch gates except mode also apply before switching.

    target_name pins a deferred request to its original target. Availability
    means fire now or switch now and recheck that exact request next step.
    """
    plane = state.get(side, {}).get(own_name, {})
    limits, weapon = profile["limits"], profile["weapon"]
    visible = {name: target for name, target in encoder.visible_enemies(state, side, own_name).items() if alive(target)}
    distances = {name: offset(plane, target, profile["units"]["feet_to_meters"],
                              profile["units"]["earth_meters_per_degree"])[3] for name, target in visible.items()}
    ordered = sorted(visible, key=lambda name: (distances[name], name))
    nearest = ordered[0] if ordered else None
    in_range = digits_mask(plane.get("TargetEnterAttackRange", 0), encoder.num_agents)
    current_mode = int(plane.get("AimMode", -1))
    reserved = _amraam_targets(state, own_name)
    specifications = {
        "sraam": ("SRAAM", int(profile["aircraft"]["sraam_capacity"]), int(weapon["aim_mode_short"]),
                  min(limits["sraam_launch_m"], limits["acm_lock_m"])),
        "amraam": ("AMRAAM", int(profile["aircraft"]["amraam_capacity"]), int(weapon["aim_mode_long"]),
                   min(limits["amraam_launch_m"], limits["crm_lock_m"])),
    }
    plans = []
    for choice in WEAPON_CHOICES:
        target = target_name if target_name is not None else nearest
        launch, switch, available = 0, 0, choice == "none"
        if alive(plane) and choice in specifications:
            prefix, capacity, mode, distance_limit = specifications[choice]
            ammo = int(plane.get(prefix+"CurrentNum", 0))
            ready = _ready_slots(plane, prefix, capacity, ammo)
            if ammo > 0 and current_mode in (int(weapon["aim_mode_short"]), int(weapon["aim_mode_long"])):
                candidates = [name for name in ordered if distances[name] <= distance_limit and
                              in_range[int(name.rsplit("_", 1)[1])] and
                              (target_name is None or name == target_name) and
                              (choice != "amraam" or name not in reserved)]
                if weapon["require_lock"]:
                    if choice == "sraam":
                        locked = _locked_indices(plane.get("SRAAMTargetLocked", 9), 1)
                    else:
                        locks = _locked_indices(plane.get("AMRAAMlockedTarget", 9999), capacity)
                        # ZK checks the next ready rail's lock first, then
                        # searches lower rail indices in descending order.
                        locked = []
                        if ready:
                            for index in range(ready[0], -1, -1):
                                if index < len(locks) and locks[index] < encoder.num_agents:
                                    locked = [locks[index]]
                                    break
                    candidates = [name for name in candidates if int(name.rsplit("_", 1)[1]) in locked]
                available = bool(ready and candidates)
                if available:
                    target = candidates[0]
                    switch = int(current_mode != mode)
                    launch = int(current_mode == mode)
        select = 10**int(target.rsplit("_", 1)[1]) if target and alive(plane) else 9
        plans.append({"available": bool(available), "fcs/weapon-launch": launch,
                      "switch-missile": switch, "change-target": select, "target": target})
    return plans


class WeaponController:
    """One deferred weapon request per aircraft, consumed on the very next step."""

    def __init__(self, encoder, profile):
        self.encoder = encoder
        self.profile = profile
        self.pending = {}

    def reset(self):
        self.pending.clear()

    def command(self, state, side, own_name, choice, enabled=True):
        request = self.pending.pop(own_name, None)
        if not enabled or not alive(state.get(side, {}).get(own_name, {})):
            choice, request = WEAPON_CHOICES.index("none"), None
        if request is not None:
            plan = weapon_plans(state, side, own_name, self.encoder, self.profile,
                                target_name=request["target"])[request["choice"]]
            # Do not retry switching, change targets, or execute the new Actor
            # weapon choice when the deferred request has become invalid.
            if not plan["available"] or plan["switch-missile"]:
                plan = {**plan, "available": False, "fcs/weapon-launch": 0, "switch-missile": 0}
            return plan
        plan = weapon_plans(state, side, own_name, self.encoder, self.profile)[int(choice)]
        if plan["available"] and plan["switch-missile"]:
            self.pending[own_name] = {"choice": int(choice), "target": plan["target"]}
        return plan
