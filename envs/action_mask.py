"""Legal multi-discrete choices, shared by sampling and PPO evaluation."""

import numpy as np

from .observation_encoder import alive
from .weapon_control import weapon_plans
from algorithms.utils.action_spec import ACTION_WIDTH, ACTION_OFFSETS, NEUTRAL_ACTION, WEAPON_OFFSET

def masks_for_side(state, side, encoder, profile, pending_requests=None):
    masks = np.ones((encoder.num_agents, ACTION_WIDTH), dtype=np.float32)
    for index in range(encoder.num_agents):
        own_name = f"{side}_{index}"
        plane = state.get(side, {}).get(own_name, {})
        if not alive(plane):
            masks[index] = 0
            for head, choice in enumerate(NEUTRAL_ACTION):
                masks[index, ACTION_OFFSETS[head] + choice] = 1
            continue
        if pending_requests is not None and own_name in pending_requests:
            # The previous step owns this weapon decision. Keep flight heads
            # trainable; the one-choice weapon head has zero log-prob/entropy.
            masks[index, WEAPON_OFFSET:] = 0
            masks[index, WEAPON_OFFSET] = 1
            continue
        plans = weapon_plans(state, side, own_name, encoder, profile)
        masks[index, WEAPON_OFFSET:] = [float(plan["available"]) for plan in plans]
    return masks
