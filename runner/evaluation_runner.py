"""Fixed-initialization red/blue swapped evaluation."""

import numpy as np
import torch

from envs.observation_encoder import alive
from envs.utils.utils import decode_aircraft_id


class EvaluationRunner:
    def __init__(self, env, actor, opponent, normalizer, opponent_normalizer=None, device="cpu"):
        self.env = env
        self.actor = actor.to(device).eval()
        self.opponent = opponent.to(device).eval()
        self.normalizer = normalizer
        self.opponent_normalizer = opponent_normalizer if opponent_normalizer is not None else normalizer
        self.device = device

    def _action(self, actor, normalizer, obs, hidden, reset_mask, action_mask):
        with torch.no_grad():
            result, _, _, next_hidden = actor(
                torch.as_tensor(normalizer.normalize(obs), dtype=torch.float32, device=self.device),
                torch.as_tensor(hidden, dtype=torch.float32, device=self.device),
                torch.as_tensor(reset_mask, dtype=torch.float32, device=self.device),
                torch.as_tensor(action_mask, dtype=torch.bool, device=self.device),
                deterministic=True,
            )
        return result.cpu().numpy(), next_hidden.cpu().numpy()

    def run_round(self, initialization, side):
        _, _, info = self.env.reset(initialization)
        other = "blue" if side == "red" else "red"
        agents = self.env.num_agents
        hidden = {side: np.zeros((agents, self.actor.hidden_dim), dtype=np.float32),
                  other: np.zeros((agents, self.opponent.hidden_dim), dtype=np.float32)}
        masks = {"red": np.zeros((agents, 1), dtype=np.float32),
                 "blue": np.zeros((agents, 1), dtype=np.float32)}
        initial_ammo = {team: [int(info["raw_state"][team][f"{team}_{i}"].get("SRAAMCurrentNum", 0)) +
                               int(info["raw_state"][team][f"{team}_{i}"].get("AMRAAMCurrentNum", 0))
                               for i in range(agents)] for team in ("red", "blue")}
        launched = {"red": 0, "blue": 0}
        hits = {"red": 0, "blue": 0}
        seen_hits = set()
        simulation = self.env.profile["simulation"]
        max_steps = max(
            int(simulation["evaluation_safety_steps"]),
            int((self.env.profile["limits"]["battle_seconds"] +
                 simulation["max_missile_overtime_seconds"]) /
                simulation["decision_seconds_fallback"]) + int(simulation["evaluation_step_margin"]),
        )
        for _ in range(max_steps):
            commands = {}
            for team, model, normalizer in ((side, self.actor, self.normalizer),
                                            (other, self.opponent, self.opponent_normalizer)):
                command, hidden[team] = self._action(model, normalizer, info["side_observations"][team], hidden[team],
                                                     masks[team], info["side_action_masks"][team])
                commands[team] = command
            _, _, _, _, info = self.env.step(commands)
            for team in ("red", "blue"):
                for i in range(agents):
                    plane = info["raw_state"][team][f"{team}_{i}"]
                    now = int(plane.get("SRAAMCurrentNum", 0)) + int(plane.get("AMRAAMCurrentNum", 0))
                    launched[team] += max(0, initial_ammo[team][i]-now)
                    initial_ammo[team][i] = now
                    masks[team][i, 0] = float(alive(plane))
                    hidden[team][i] *= masks[team][i, 0]
            for missile_name, missile in info["raw_state"].get("missile", {}).items():
                if missile_name in seen_hits or int(missile.get("Status", 0)) != 1:
                    continue
                seen_hits.add(missile_name)
                owner = decode_aircraft_id(missile.get("Owner"))
                if owner:
                    hits[owner.split("_", 1)[0]] += 1
            if info["episode_done"]:
                winner = info["outcome"]
                return {"side": side, "winner": winner, "result": "tie" if winner == "tie" else
                        "win" if winner == side else "loss", "red_alive": info["red_score"][0],
                        "blue_alive": info["blue_score"][0], "red_missiles": info["red_score"][1],
                        "blue_missiles": info["blue_score"][1], "red_launched": launched["red"],
                        "blue_launched": launched["blue"], "red_hits": hits["red"],
                        "blue_hits": hits["blue"], "sim_time": info["sim_time"]}
        raise RuntimeError("Evaluation round did not terminate within the configured safety limit")
