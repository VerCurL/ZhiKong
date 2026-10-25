"""Recurrent MAPPO update with legal-action and active-agent masking."""

import torch
from torch import nn


class PPOTrainer:
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config
        self.device = policy.device

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def update(self, buffer, rng, progress=None):
        config = self.config
        totals = {key: 0.0 for key in ("policy_loss", "value_loss", "entropy", "ratio")}
        count = 0
        actor_count = 0
        self.policy.actor.train()
        self.policy.critic.train()
        for epoch in range(int(config["ppo_epochs"])):
            for batch in buffer.recurrent_batches(config["sequence_length"], config["mini_batches"], rng):
                obs = self._tensor(batch["obs"])
                state = self._tensor(batch["state"])
                actions = self._tensor(batch["actions"], torch.long)
                action_masks = self._tensor(batch["action_masks"], torch.bool)
                resets = self._tensor(batch["reset_masks"])
                active = self._tensor(batch["active_masks"])
                policy_active = active*self._tensor(batch["policy_control_masks"])
                old_log_probs = self._tensor(batch["old_log_probs"])
                old_values = self._tensor(batch["old_values"])
                returns = self._tensor(batch["returns"])
                advantages = self._tensor(batch["advantages"])
                actor_hidden = self._tensor(batch["actor_hidden"])
                critic_hidden = self._tensor(batch["critic_hidden"])
                values, _ = self.policy.critic(state, critic_hidden, resets)
                denominator = active.sum().clamp(min=1.0)
                has_policy = bool(policy_active.sum().item() > 0)
                policy_loss = entropy_mean = ratio_mean = torch.zeros((), device=self.device)
                if has_policy:
                    new_log_probs, entropy = self.policy.actor.evaluate_actions(
                        obs, actor_hidden, resets, action_masks, actions)
                    log_ratio = torch.where(policy_active > 0, new_log_probs-old_log_probs, torch.zeros_like(new_log_probs))
                    ratio = torch.exp(log_ratio)
                    clipped = torch.clamp(ratio, 1-config["clip_param"], 1+config["clip_param"])
                    policy_denominator = policy_active.sum()
                    policy_loss = -(torch.minimum(ratio*advantages, clipped*advantages)*policy_active).sum()/policy_denominator
                    entropy_mean = (entropy*policy_active).sum()/policy_denominator
                    ratio_mean = (ratio*policy_active).sum()/policy_denominator
                    actor_count += 1
                value_error = (values-returns).square()
                if config.get("use_clipped_value_loss", False):
                    clipped_value = old_values+(values-old_values).clamp(-config["clip_param"], config["clip_param"])
                    value_error = torch.maximum(value_error, (clipped_value-returns).square())
                value_loss = (value_error*active).sum()/(2*denominator)
                loss = policy_loss + config["value_loss_coef"]*value_loss - config["entropy_coef"]*entropy_mean
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), config["max_grad_norm"])
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), config["max_grad_norm"])
                self.policy.optimizer.step()
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy_mean.item())
                totals["ratio"] += float(ratio_mean.item())
                count += 1
            if progress is not None:
                progress(epoch+1, int(config["ppo_epochs"]))
        self.policy.actor.eval()
        self.policy.critic.eval()
        return {**{key: value/max(actor_count if key != "value_loss" else count, 1) for key, value in totals.items()},
                "optimizer_updates": count, "actor_optimizer_updates": actor_count}
