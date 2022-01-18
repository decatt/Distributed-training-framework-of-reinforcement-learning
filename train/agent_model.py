import torch.nn as nn
import torch
import numpy as np
from tools import CategoricalMasked
import netron

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MobaNet(nn.Module):
    def __init__(self, action_length=178):
        super(MobaNet, self).__init__()
        self.version = 0

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            # layer_init(nn.Linear(32 * 6 * 6, 256)),
            layer_init(nn.Linear(32 * 3 * 3, 256)),
            nn.ReLU(), )

        self.actor = layer_init(nn.Linear(256, action_length), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        obs = self.network(x)
        return self.actor(obs), self.critic(obs)


class MobaAgent:
    def __init__(self, net: nn.Module, action_space: list):
        self.net = net
        self.action_space = action_space

    def get_log_prob_entropy_value(self, x, action, masks):
        logits, value = self.net(x)
        split_logits = torch.split(logits, self.action_space, dim=1)
        split_masks = torch.split(masks, self.action_space, dim=1)
        multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_masks)]
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return log_prob.sum(0), entropy.sum(0), value

    def get_value(self, x):
        return self.net(x)[1]


if __name__ == '__main__':
    netron.start('moba_net.onnx')

