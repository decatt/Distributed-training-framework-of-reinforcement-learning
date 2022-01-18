import random
import numpy as np
import torch
from torch.distributions.categorical import Categorical

use_gpu = False
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    # H = sum(p(x)log(p(x)))
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward: float, next_state, done, mask, logit, version):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, mask, logit, version)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, masks, logits = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, masks, logits

    def __len__(self):
        return len(self.buffer)


# state, actions, reward, values, next_state, done, masks, log_probs, version
def get_replays_from_redis(batch_size: int) -> tuple:
    pass
