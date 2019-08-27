from torch.nn import functional as F
import torch


def zoneout1(new_state, state, zoneout_prob, is_training):
    """As in a tensorflow implementation."""
    if is_training:
        new_state = F.dropout(new_state - state, zoneout_prob) + state
    else:
        new_state = zoneout_prob * state + (1 - zoneout_prob) * new_state
    return new_state


def zoneout2(new_state, state, zoneout_prob, is_training):
    """As in the original paper."""
    if is_training:
        keep_new = torch.floor(torch.zeros_like(state).uniform_()
                                     + (1.0 - zoneout_prob))
        keep_old = 1.0 - keep_new
        new_state = keep_new * new_state + keep_old * state
    else:
        new_state = zoneout_prob * state + (1 - zoneout_prob) * new_state
    return new_state