"""Actor/Critic ownership, inference and checkpoint state."""

import torch

from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic


class PPOPolicy:
    def __init__(self, obs_dim, state_dim, config, device="cpu"):
        self.device = torch.device(device)
        self.actor = PPOActor(obs_dim, int(config["actor_hidden"])).to(self.device)
        self.critic = PPOCritic(state_dim, int(config["critic_hidden"])).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(config["learning_rate"]),
        )

    def act(self, obs, state, actor_hidden, critic_hidden, reset_mask, action_mask, deterministic=False):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            actor_hidden = torch.as_tensor(actor_hidden, dtype=torch.float32, device=self.device)
            critic_hidden = torch.as_tensor(critic_hidden, dtype=torch.float32, device=self.device)
            reset_mask = torch.as_tensor(reset_mask, dtype=torch.float32, device=self.device)
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            action, log_prob, _, next_actor = self.actor(obs, actor_hidden, reset_mask, action_mask, deterministic)
            value, next_critic = self.critic(state, critic_hidden, reset_mask)
        return tuple(item.cpu().numpy() for item in (action, log_prob, value, next_actor, next_critic))

    def value(self, state, hidden, reset_mask):
        with torch.no_grad():
            value, _ = self.critic(
                torch.as_tensor(state, dtype=torch.float32, device=self.device),
                torch.as_tensor(hidden, dtype=torch.float32, device=self.device),
                torch.as_tensor(reset_mask, dtype=torch.float32, device=self.device),
            )
        return value.cpu().numpy()

    def checkpoint(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict()}

    def load_checkpoint(self, checkpoint, optimizer=False):
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
