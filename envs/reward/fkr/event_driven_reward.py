"""Private outcome ledger; never supplies hidden data to dense components."""

from ...observation_encoder import alive
from ...utils.utils import decode_aircraft_id
from ...termination import outcome


EVENT_NAMES = ("terminal", "kill", "death", "hit", "miss", "evade", "damage")
EVENT_VALUES = {"enemy_death": 6.25, "own_death": -6.25, "hit": 2.5, "miss": -.2,
                "evade": .5, "damage": -2.5, "win": 100.0, "loss": -100.0, "tie": 0.0}
MISS_STATUSES = (2, 3, 4, 5)
EVADE_STATUSES = (2, 3)


class EventDrivenReward:
    def __init__(self, life_max):
        self.settings, self.life_max = dict(EVENT_VALUES), life_max
        self.reset({})

    def reset(self, state):
        self.aircraft = {side: {name: (alive(plane), float(plane.get("LifeCurrent", self.life_max)))
                                for name, plane in state.get(side, {}).items()} for side in ("red", "blue")}
        self.dead = set()
        self.terminal_paid = False
        self.missiles = {}
        for name, missile in state.get("missile", {}).items():
            if isinstance(missile, dict):
                self.missiles[name] = self._record(missile)

    @staticmethod
    def _record(missile):
        status = int(missile.get("Status", 0))
        return {"status": status, "owner": decode_aircraft_id(missile.get("Owner")),
                "target": decode_aircraft_id(missile.get("Target")), "settled": status != 0}

    def compute(self, state, terminal=False):
        result = {side: dict.fromkeys(EVENT_NAMES, 0.0) for side in ("red", "blue")}
        for side in ("red", "blue"):
            other = "blue" if side == "red" else "red"
            for name, plane in state.get(side, {}).items():
                before = self.aircraft[side].get(name)
                hp = float(plane.get("LifeCurrent", before[1] if before else self.life_max))
                living = alive(plane)
                if before is not None:
                    result[side]["damage"] += self.settings["damage"]*max(0.0, before[1]-hp)/self.life_max
                    if before[0] and not living and name not in self.dead:
                        result[side]["death"] += self.settings["own_death"]
                        result[other]["kill"] += self.settings["enemy_death"]
                        self.dead.add(name)
                self.aircraft[side][name] = (living, hp)
        for name, missile in state.get("missile", {}).items():
            if not isinstance(missile, dict):
                continue
            status = int(missile.get("Status", 0))
            before = self.missiles.get(name)
            if before is None or (before["settled"] and status == 0):
                self.missiles[name] = self._record(missile)
                continue
            if before["settled"]:
                continue
            owner = decode_aircraft_id(missile.get("Owner")) or before["owner"]
            target = decode_aircraft_id(missile.get("Target")) or before["target"]
            before.update(owner=owner, target=target)
            if status != 0 and before["status"] == 0:
                if owner:
                    side = owner.split("_", 1)[0]
                    if status == 1:
                        result[side]["hit"] += self.settings["hit"]
                    elif status in MISS_STATUSES:
                        result[side]["miss"] += self.settings["miss"]
                if status in EVADE_STATUSES and owner and target:
                    side = target.split("_", 1)[0]
                    if not owner.startswith(side+"_") and alive(state.get(side, {}).get(target, {})):
                        result[side]["evade"] += self.settings["evade"]
                before["settled"] = True
            before["status"] = status
        if terminal and not self.terminal_paid:
            winner = outcome(state)
            for side in result:
                result[side]["terminal"] = self.settings["tie" if winner == "tie" else "win" if winner == side else "loss"]
            self.terminal_paid = True
        return result
