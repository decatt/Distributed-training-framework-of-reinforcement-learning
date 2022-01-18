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

    def get_action(self, x, type_masks=None, envs=None):
        logits, value = self.net(x)
        split_logits = torch.split(logits, self.action_space, dim=1)
        type_masks = torch.Tensor(type_masks)
        multi_categoricals = [CategoricalMasked(logits=split_logits[0], masks=type_masks)]
        action_components = [multi_categoricals[0].sample()]
        action_masks = np.array(envs.vec_client.getUnitActionMasks(action_components[0].cpu().numpy())).reshape(10, -1)
        split_suam = torch.split(torch.Tensor(action_masks), self.action_space[1:], dim=1)
        multi_categoricals = multi_categoricals + [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                                   zip(split_logits[1:], split_suam)]
        masks = torch.cat((type_masks, torch.Tensor(action_masks)), 1)
        action_components += [categorical.sample() for categorical in multi_categoricals[1:]]
        action = torch.stack(action_components)
        log_prob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        return action.numpy(), log_prob.sum(0).numpy().reshape(-1), masks.numpy(), value.numpy().reshape(-1)

    def get_value(self, x):
        return self.net(x)[1]

    def get_grads(self):
        grads = []
        for p in self.net.parameters():
            grads.append(p.grad)
        return grads


if __name__ == '__main__':
    dummy_input = torch.randn((1000,10,10,27))
    model = MobaNet()
    torch.onnx.export(model, dummy_input, "moba_net.onnx")
    netron.start('moba_net.onnx')
