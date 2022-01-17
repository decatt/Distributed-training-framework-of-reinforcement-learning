import torch.nn as nn
import torch
import numpy as np
import torchvision
from tools import CategoricalMasked
import netron

device = torch.device('cuda:0' if torch.cuda.is_available() and True else 'cpu')


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MobaNet(nn.Module):
    def __init__(self, state_length=1000, action_length=30):
        super(MobaNet, self).__init__()
        self.version = 0

        self.network = nn.Sequential(
            layer_init(nn.Linear(state_length, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
        )

        self.actor = layer_init(nn.Linear(256, action_length), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        obs = self.network(x)
        return self.actor(obs), self.critic(obs)


class MobaAgent:
    def __init__(self, net: nn.Module, action_space: list):
        self.net = net
        self.action_space = action_space

    def get_action(self, x, type_masks=None, cast_masks=None, action=None, masks=None):
        logits, value = self.net(x).cpu()
        split_logits = torch.split(logits, self.action_space, dim=1)

        if action is None:
            type_masks = torch.Tensor(type_masks)
            multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=type_masks)]
            action_components = [multi_categoricals[0].sample()]
            action_masks = []
            for i in range(len(type_masks)):
                action_masks.append(cast_masks[i][action_components[i]])
            split_suam = torch.split(torch.Tensor(action_masks), self.action_space[1:], dim=1)
            multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                                       zip(split_logits[1:], split_suam)]
            masks = torch.cat((type_masks, torch.Tensor(action_masks)), 1)
            action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
            action = torch.stack(action_components)
        else:
            split_masks = torch.split(masks, self.action_space, dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_masks)]
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, log_prob.sum(0), entropy.sum(0), masks, value

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

    def get_grads(self):
        grads = []
        for p in self.net.parameters():
            grads.append(p.grad)
        return grads


if __name__ == '__main__':
    dummy_input = torch.randn((1000,1000))
    model = MobaNet()
    torch.onnx.export(model, dummy_input, "moba_net.onnx")
    netron.start('moba_net.onnx')

