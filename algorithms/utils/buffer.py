"""Time-major on-policy buffer adapted from LAGCFC's SharedReplayBuffer."""

import numpy as np

from .action_spec import ACTION_DIMS, ACTION_WIDTH


class SharedReplayBuffer:
    def __init__(self, steps, agents, obs_dim, state_dim, actor_hidden, critic_hidden,
                 gamma, gae_lambda, action_dim=None, action_width=ACTION_WIDTH):
        self.steps = int(steps)
        self.agents = int(agents)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        shape = (self.steps, self.agents)
        self.obs = np.zeros((self.steps+1, self.agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((self.steps+1, self.agents, state_dim), dtype=np.float32)
        self.action_dim = int(action_dim if action_dim is not None else len(ACTION_DIMS))
        self.action_width = int(action_width)
        self.actions = np.zeros((*shape, self.action_dim), dtype=np.int64)
        self.action_masks = np.zeros((*shape, self.action_width), dtype=bool)
        self.rewards = np.zeros((*shape, 1), dtype=np.float32)
        self.values = np.zeros((self.steps+1, self.agents, 1), dtype=np.float32)
        self.log_probs = np.zeros((*shape, 1), dtype=np.float32)
        self.reset_masks = np.ones((self.steps+1, self.agents, 1), dtype=np.float32)
        self.active_masks = np.ones((self.steps+1, self.agents, 1), dtype=np.float32)
        self.policy_control_masks = np.ones((self.steps, self.agents, 1), dtype=np.float32)
        self.actor_hidden = np.zeros((self.steps+1, self.agents, actor_hidden), dtype=np.float32)
        self.critic_hidden = np.zeros((self.steps+1, self.agents, critic_hidden), dtype=np.float32)
        self.returns = np.zeros_like(self.values)
        self.advantages = np.zeros_like(self.rewards)
        self.cursor = 0

    def start(self, obs, state, actor_hidden, critic_hidden, reset_mask, active_mask):
        self.cursor = 0
        self.obs[0] = obs
        self.state[0] = state
        self.actor_hidden[0] = actor_hidden
        self.critic_hidden[0] = critic_hidden
        self.reset_masks[0] = reset_mask
        self.active_masks[0] = active_mask

    def insert(self, actions, action_mask, rewards, value, log_prob, next_obs, next_state,
               next_actor_hidden, next_critic_hidden, next_reset_mask, next_active_mask, policy_control_mask=None):
        t = self.cursor
        if t >= self.steps:
            raise RuntimeError("Rollout buffer is full")
        self.actions[t] = actions
        self.policy_control_masks[t] = self.active_masks[t] if policy_control_mask is None else policy_control_mask
        self.action_masks[t] = action_mask
        self.rewards[t] = rewards
        self.values[t] = value
        self.log_probs[t] = log_prob
        self.obs[t+1] = next_obs
        self.state[t+1] = next_state
        self.actor_hidden[t+1] = next_actor_hidden
        self.critic_hidden[t+1] = next_critic_hidden
        self.reset_masks[t+1] = next_reset_mask
        self.active_masks[t+1] = next_active_mask
        self.cursor += 1

    def compute_returns(self, bootstrap):
        if self.cursor != self.steps:
            raise RuntimeError("Rollout must be full before GAE calculation")
        self.values[-1] = bootstrap
        gae = np.zeros((self.agents, 1), dtype=np.float32)
        for t in range(self.steps-1, -1, -1):
            continuation = self.reset_masks[t+1]
            delta = self.rewards[t] + self.gamma*self.values[t+1]*continuation - self.values[t]
            gae = delta + self.gamma*self.gae_lambda*continuation*gae
            self.returns[t] = gae + self.values[t]
        active = self.active_masks[:-1]*self.policy_control_masks
        raw = self.returns[:-1]-self.values[:-1]
        count = max(float(active.sum()), 1.0)
        mean = float((raw*active).sum()/count)
        variance = float((((raw-mean)**2)*active).sum()/count)
        self.advantages = ((raw-mean)/(np.sqrt(variance)+1e-5))*active

    def recurrent_batches(self, sequence_length, mini_batches, rng):
        length = int(sequence_length)
        if self.steps % length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        chunks = [(agent, start) for agent in range(self.agents)
                  for start in range(0, self.steps, length)]
        rng.shuffle(chunks)
        if len(chunks) < mini_batches:
            raise ValueError("More mini-batches than recurrent chunks")
        for selected in np.array_split(np.arange(len(chunks)), mini_batches):
            keys = [chunks[int(index)] for index in selected]
            def sequence(array):
                return np.stack([array[start:start+length, agent] for agent, start in keys], axis=1)
            yield {
                "obs": sequence(self.obs),
                "state": sequence(self.state),
                "actions": sequence(self.actions),
                "action_masks": sequence(self.action_masks),
                "reset_masks": sequence(self.reset_masks),
                "active_masks": sequence(self.active_masks),
                "policy_control_masks": sequence(self.policy_control_masks),
                "old_log_probs": sequence(self.log_probs),
                "old_values": sequence(self.values),
                "returns": sequence(self.returns),
                "advantages": sequence(self.advantages),
                "actor_hidden": np.stack([self.actor_hidden[start, agent] for agent, start in keys]),
                "critic_hidden": np.stack([self.critic_hidden[start, agent] for agent, start in keys]),
            }


class CompleteEpisodeBuffer:
    """RAM-only variable-length episodes; chunks never cross episode boundaries.

    Store recurrent states only at chunk starts, rather than at every timestep.
    Padded tails have active_masks=0 and are excluded from GAE and every loss.
    """

    def __init__(self, agents, obs_dim, state_dim, actor_hidden, critic_hidden, config):
        self.agents = agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.actor_hidden_dim = actor_hidden
        self.critic_hidden_dim = critic_hidden
        self.length = int(config["sequence_length"])
        self.gamma = float(config["gamma"])
        self.gae_lambda = float(config["gae_lambda"])
        self.max_batch_samples = int(config["max_minibatch_samples"])
        self.memory_limit = int(config["sampling_memory_limit_mb"]*1024**2)
        self.chunks = []
        self.open_episodes = {}
        self.completed_episodes = 0
        self.environment_steps = 0
        self.nbytes = 0
        self.sealed = False

    def begin_episode(self, env_id):
        if self.sealed or env_id in self.open_episodes:
            raise RuntimeError("Episode already open or collection sealed")
        self.open_episodes[env_id] = []

    def _new_chunk(self, actor_hidden, critic_hidden):
        shape = (self.length, self.agents)
        widths = {"obs": self.obs_dim, "state": self.state_dim, "rewards": 1,
                  "old_log_probs": 1, "old_values": 1, "returns": 1, "advantages": 1,
                  "reset_masks": 1, "next_reset_masks": 1, "active_masks": 1, "policy_control_masks": 1}
        estimated = self.length*self.agents*(sum(widths.values())*4 + len(ACTION_DIMS)*8 + ACTION_WIDTH)
        estimated += self.agents*(self.actor_hidden_dim+self.critic_hidden_dim)*4
        if self.nbytes+estimated > self.memory_limit:
            raise MemoryError("Sampling RAM limit reached; reduce n_rollout_threads (total battles per update)")
        chunk = {name: np.zeros((*shape, width), dtype=np.float32) for name, width in widths.items()}
        chunk["actions"] = np.zeros((*shape, len(ACTION_DIMS)), dtype=np.int64)
        # Padding must also be a valid categorical input, even though its loss is zero.
        chunk["action_masks"] = np.ones((*shape, ACTION_WIDTH), dtype=bool)
        chunk["actor_hidden"] = np.asarray(actor_hidden, dtype=np.float32).copy()
        chunk["critic_hidden"] = np.asarray(critic_hidden, dtype=np.float32).copy()
        chunk["length"] = 0
        self.nbytes += sum(value.nbytes for value in chunk.values() if isinstance(value, np.ndarray))
        self.chunks.append(chunk)
        return chunk

    def append(self, env_id, *, obs, state, actions, action_masks, rewards, old_log_probs,
               old_values, reset_masks, next_reset_masks, active_masks, policy_control_masks, actor_hidden, critic_hidden):
        if self.sealed or env_id not in self.open_episodes:
            raise RuntimeError("Cannot append outside an open collection episode")
        blocks = self.open_episodes[env_id]
        if not blocks or blocks[-1]["length"] == self.length:
            blocks.append(self._new_chunk(actor_hidden, critic_hidden))
        chunk = blocks[-1]
        step = chunk["length"]
        for name, value in (("obs", obs), ("state", state), ("actions", actions),
                            ("action_masks", action_masks), ("rewards", rewards),
                            ("old_log_probs", old_log_probs), ("old_values", old_values),
                            ("reset_masks", reset_masks), ("next_reset_masks", next_reset_masks),
                            ("active_masks", active_masks), ("policy_control_masks", policy_control_masks)):
            chunk[name][step] = value
        chunk["length"] += 1
        self.environment_steps += 1

    def finish_episode(self, env_id):
        blocks = self.open_episodes[env_id]
        if not blocks or np.any(blocks[-1]["next_reset_masks"][blocks[-1]["length"]-1]):
            raise RuntimeError("Only genuinely terminated episodes may enter training")
        gae = np.zeros((self.agents, 1), dtype=np.float32)
        next_value = np.zeros_like(gae)
        for chunk in reversed(blocks):
            for step in range(chunk["length"]-1, -1, -1):
                continuation = chunk["next_reset_masks"][step]
                value = chunk["old_values"][step]
                delta = chunk["rewards"][step]+self.gamma*next_value*continuation-value
                gae = delta+self.gamma*self.gae_lambda*continuation*gae
                chunk["returns"][step] = gae+value
                chunk["advantages"][step] = gae
                next_value = value
        del self.open_episodes[env_id]
        self.completed_episodes += 1

    def seal(self):
        if self.open_episodes or not self.completed_episodes:
            raise RuntimeError("All sampling episodes must finish before PPO training")
        count = sum(float(block["active_masks"].sum()) for block in self.chunks)
        if count <= 0:
            raise RuntimeError("Collection contains no active training samples")
        policy_count = sum(float((b["active_masks"]*b["policy_control_masks"]).sum()) for b in self.chunks)
        denominator = max(policy_count, 1.0)
        mean = sum(float((b["advantages"]*b["active_masks"]*b["policy_control_masks"]).sum(dtype=np.float64))
                   for b in self.chunks)/denominator
        variance = sum(float((((b["advantages"]-mean)**2)*b["active_masks"]*b["policy_control_masks"]).sum(dtype=np.float64))
                       for b in self.chunks)/denominator
        for block in self.chunks:
            block["advantages"][:] = ((block["advantages"]-mean)/(np.sqrt(variance)+1e-5)*
                                      block["active_masks"]*block["policy_control_masks"])
        self.sealed = True

    @property
    def training_chunks(self):
        return [(index, agent) for index, block in enumerate(self.chunks)
                for agent in range(self.agents) if np.any(block["active_masks"][:, agent])]

    def recurrent_batches(self, sequence_length, mini_batches, rng):
        if not self.sealed or sequence_length != self.length:
            raise RuntimeError("Training requires a sealed collection with matching chunk length")
        keys = self.training_chunks
        rng.shuffle(keys)
        batch_chunks = self.max_batch_samples//self.length
        groups = np.array_split(np.arange(len(keys)), min(mini_batches, len(keys)))
        selections = ([keys[int(i)] for i in group[start:start+batch_chunks]]
                      for group in groups for start in range(0, len(group), batch_chunks))
        for selected in selections:
            batch = {}
            for name in ("obs", "state", "actions", "action_masks", "reset_masks", "active_masks", "policy_control_masks",
                         "old_log_probs", "old_values", "returns", "advantages"):
                batch[name] = np.stack([self.chunks[index][name][:, agent] for index, agent in selected], axis=1)
            for name in ("actor_hidden", "critic_hidden"):
                batch[name] = np.stack([self.chunks[index][name][agent] for index, agent in selected])
            yield batch
