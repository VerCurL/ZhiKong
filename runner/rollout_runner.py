"""Collect a fixed battle count in parallel batches, with a barrier between batches."""

import numpy as np
import torch

from algorithms.utils.buffer import CompleteEpisodeBuffer
from algorithms.utils.normalization import RunningMoments
from algorithms.utils.torch_execution import torch_thread_count
from envs.observation_encoder import alive
from envs.formations import FORMATION_NAMES, formation_schedule
from envs.reward.fkr.multiplecombat_reward import REWARD_METRIC_COLUMNS, summarize_reward_metrics


class RolloutRunner:
    def __init__(self, envs, policy, opponent_actor, opponent_pool, normalizer, config):
        self.envs = envs
        self.policy = policy
        self.opponent_actor = opponent_actor
        self.opponent_pool = opponent_pool
        self.normalizer = normalizer
        self.config = config
        self.side = envs.side
        self.other = "blue" if self.side == "red" else "red"
        self.num_agents = envs.num_agents
        self.opponent_normalizer = RunningMoments(envs.encoder.actor_dim)
        self.spawn_rng = np.random.default_rng(config["seed"])

    def _active(self, info, side):
        return np.asarray([[float(alive(info["raw_state"][side][f"{side}_{i}"]))]
                           for i in range(self.num_agents)], dtype=np.float32)

    @torch.no_grad()
    def _opponent_actions(self, observations, hidden, reset_mask, legal):
        action, _, _, next_hidden = self.opponent_actor(
            torch.as_tensor(self.opponent_normalizer.normalize(observations), device=self.policy.device),
            torch.as_tensor(hidden, device=self.policy.device),
            torch.as_tensor(reset_mask, device=self.policy.device),
            torch.as_tensor(legal, dtype=torch.bool, device=self.policy.device),
        )
        return action.cpu().numpy(), next_hidden.cpu().numpy()

    def collect(self, progress=None):
        with torch_thread_count(self.config["torch_threads"]["sampling"]):
            return self._collect(progress)

    def _collect(self, progress=None):
        total_battles = int(self.config["n_rollout_threads"])
        if total_battles <= 0 or total_battles % 2:
            raise ValueError("n_rollout_threads must be a positive even number for equal left/right spawns")
        opening_schedule = formation_schedule(total_battles, self.side, self.spawn_rng)
        # No optimizer is called in this phase. Networks and inference statistics
        # remain fixed for the entire set of complete battles.
        self.policy.actor.eval()
        self.policy.critic.eval()
        opponent_id = self.opponent_pool.sample_into(
            self.opponent_actor, self.opponent_normalizer, self.policy.actor, self.normalizer)
        self.opponent_actor.eval()
        pending_normalizer = RunningMoments(self.envs.encoder.actor_dim)
        n, m = self.envs.num_envs, self.num_agents
        if progress is not None:
            progress(0, total_battles, opponent_id)
        buffer = CompleteEpisodeBuffer(m, self.envs.encoder.actor_dim, self.envs.encoder.critic_dim,
                                       self.policy.actor.hidden_dim, self.policy.critic.hidden_dim, self.config)
        actor_h = np.zeros((n, m, self.policy.actor.hidden_dim), dtype=np.float32)
        critic_h = np.zeros((n, m, self.policy.critic.hidden_dim), dtype=np.float32)
        opponent_h = np.zeros((n, m, self.opponent_actor.hidden_dim), dtype=np.float32)
        reset_masks = np.zeros((n, m, 1), dtype=np.float32)
        opponent_masks = np.zeros_like(reset_masks)
        frames = {}
        episode_steps = np.zeros(n, dtype=np.int64)
        episode_rewards = np.zeros(n, dtype=np.float64)
        episode_spawn_sides = {}
        active_envs = []
        episodes = []
        reward_metrics = dict.fromkeys(REWARD_METRIC_COLUMNS, 0.0)
        episode_metrics = [dict.fromkeys(REWARD_METRIC_COLUMNS, 0.0) for _ in range(n)]
        while len(episodes) < total_battles:
            if not active_envs:
                # Reuse physical environments only after the entire previous
                # batch has terminated. The last batch may use fewer workers.
                active_envs = list(range(min(n, total_battles-len(episodes))))
                openings = opening_schedule[len(episodes):len(episodes)+len(active_envs)]
                fresh = self.envs.call(active_envs, "reset_training", openings)
                for index, frame, opening in zip(active_envs, fresh, openings):
                    frames[index] = frame
                    episode_spawn_sides[index] = opening["spawn_side"]
                    actor_h[index].fill(0)
                    critic_h[index].fill(0)
                    opponent_h[index].fill(0)
                    reset_masks[index].fill(0)
                    opponent_masks[index].fill(0)
                    episode_steps[index] = 0
                    episode_rewards[index] = 0
                    episode_metrics[index] = dict.fromkeys(REWARD_METRIC_COLUMNS, 0.0)
                    buffer.begin_episode(index)
            ids = list(active_envs)
            observations = np.stack([frames[i][0] for i in ids])
            states = np.stack([frames[i][1] for i in ids])
            infos = [frames[i][2] for i in ids]
            alive_masks = np.stack([self._active(info, self.side) for info in infos])
            pending_normalizer.update(observations[alive_masks[..., 0].astype(bool)])
            normalized = self.normalizer.normalize(observations)
            legal = np.stack([info["side_action_masks"][self.side] for info in infos])
            actions, log_probs, values, next_actor, next_critic = self.policy.act(
                normalized.reshape(-1, self.envs.encoder.actor_dim),
                states.reshape(-1, self.envs.encoder.critic_dim),
                actor_h[ids].reshape(-1, self.policy.actor.hidden_dim),
                critic_h[ids].reshape(-1, self.policy.critic.hidden_dim),
                reset_masks[ids].reshape(-1, 1), legal.reshape(-1, legal.shape[-1]))
            opponent_obs = np.concatenate([info["side_observations"][self.other] for info in infos])
            opponent_legal = np.concatenate([info["side_action_masks"][self.other] for info in infos])
            opponent_actions, next_opponent = self._opponent_actions(
                opponent_obs, opponent_h[ids].reshape(-1, self.opponent_actor.hidden_dim),
                opponent_masks[ids].reshape(-1, 1), opponent_legal)
            actions, log_probs, values, next_actor, next_critic, opponent_actions, next_opponent = [
                value.reshape(len(ids), m, -1) for value in
                (actions, log_probs, values, next_actor, next_critic, opponent_actions, next_opponent)]
            commands = [{self.side: actions[j], self.other: opponent_actions[j]} for j in range(len(ids))]
            results = self.envs.call(ids, "step", commands)
            previous_completed = len(episodes)
            for j, index in enumerate(ids):
                next_obs, next_state, rewards, dones, info = results[j]
                for key, value in info["reward_detail"]["team"][self.side]["metrics"].items():
                    reward_metrics[key] += value
                    episode_metrics[index][key] += value
                episode_steps[index] += 1
                episode_rewards[index] += float(rewards.mean())
                continuation = (~dones).astype(np.float32).reshape(m, 1)
                if info["episode_done"]:
                    continuation.fill(0)
                buffer.append(index, obs=normalized[j], state=states[j], actions=actions[j],
                              action_masks=legal[j], rewards=rewards, old_log_probs=log_probs[j],
                              old_values=values[j], reset_masks=reset_masks[index],
                              next_reset_masks=continuation, active_masks=alive_masks[j],
                              policy_control_masks=info["policy_control_masks"][self.side],
                              actor_hidden=actor_h[index], critic_hidden=critic_h[index])
                actor_h[index] = next_actor[j]*continuation
                critic_h[index] = next_critic[j]*continuation
                reset_masks[index] = continuation
                opponent_masks[index] = self._active(info, self.other)
                opponent_h[index] = next_opponent[j]*opponent_masks[index]
                frames[index] = (next_obs, next_state, info)
                if info["episode_done"]:
                    buffer.finish_episode(index)
                    episodes.append({"env_id": index, "outcome": info["outcome"],
                                     "red_score": info["red_score"], "blue_score": info["blue_score"],
                                     "sim_time": info["sim_time"], "steps": int(episode_steps[index]),
                                     "reward": float(episode_rewards[index]), "spawn_side": episode_spawn_sides[index],
                                     "formations": dict(info["formations"]),
                                     "reward_metrics": episode_metrics[index].copy()})
                    active_envs.remove(index)
                elif episode_steps[index] >= int(self.config["collection_max_episode_steps"]):
                    raise RuntimeError(f"Environment {index} exceeded the episode safety limit; partial data will not train")
            if progress is not None and len(episodes) != previous_completed:
                progress(len(episodes), total_battles, opponent_id)
        buffer.seal()
        return buffer, {"reward_sum": sum(episode["reward"] for episode in episodes),
                        "left_spawn_rounds": sum(episode["spawn_side"] == "left" for episode in episodes),
                        "right_spawn_rounds": sum(episode["spawn_side"] == "right" for episode in episodes),
                        "formation_counts": {
                            **{f"formation_{name}_rounds": sum(e["formations"][self.side] == name for e in episodes) for name in FORMATION_NAMES},
                            **{f"opponent_formation_{name}_rounds": sum(e["formations"][self.other] == name for e in episodes) for name in FORMATION_NAMES}},
                        "episodes": episodes, "pending_normalizer": pending_normalizer,
                        "opponent": opponent_id, "environment_steps": buffer.environment_steps,
                        "sample_slots": buffer.environment_steps*m, "training_chunks": len(buffer.training_chunks),
                        "buffer_mb": buffer.nbytes/1024**2,
                        "reward_metrics": summarize_reward_metrics(reward_metrics)}
