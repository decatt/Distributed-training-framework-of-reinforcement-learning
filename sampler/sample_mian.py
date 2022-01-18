import torch
import numpy as np
from gym_microrts.envs.vec_env import MicroRTSVecEnv
from gym_microrts import microrts_ai
from agent import MobaNet, MobaAgent


def init_seeds(torch_seed=0, seed=0):
    torch.manual_seed(torch_seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(torch_seed)  # Sets the seed for generating random numbers for the current GPU.
    torch.cuda.manual_seed_all(torch_seed)  # Sets the seed for generating random numbers on all GPUs.
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    num_envs = 10
    num_steps = 1024

    device = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu')

    ais = []
    for i in range(num_envs):
        ais.append(microrts_ai.coacAI)

    init_seeds()
    envs = MicroRTSVecEnv(
        num_envs=num_envs,
        max_steps=10000,
        render_theme=2,
        ai2s=ais,
        frame_skip=10,
        map_path="maps/10x10/basesWorkers10x10.xml",
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])
    )

    next_obs = envs.reset()
    next_done = np.zeros(num_envs)
    starting_update = 1
    gae_lambda = 0.95
    num_updates = 1

    obs_space = (10, 10, 27)
    action_shape = [100, 6, 4, 4, 4, 4, 7, 49]

    states = np.ones((num_steps, num_envs, ) + obs_space)
    actions = np.ones((num_steps, num_envs, len(action_shape)))
    rewards = np.ones((num_steps, num_envs))
    values = np.ones((num_steps, num_envs))
    ds = np.ones((num_steps, num_envs))
    masks = np.ones((num_steps, num_envs, sum(action_shape)))
    log_probs = np.ones((num_steps, num_envs))

    moba_net = MobaNet(action_length=sum(action_shape))
    moba_agent = MobaAgent(moba_net, action_space=action_shape)

    for update in range(starting_update, num_updates + 1):
        path = ''

        for step in range(0, num_steps):
            envs.render()
            states[step] = next_obs
            ds[step] = next_done
            unit_mask = np.array(envs.vec_client.getUnitLocationMasks()).reshape(num_envs, -1)
            with torch.no_grad():
                action, log_prob, masks[step], values[step] = moba_agent.get_action(torch.Tensor(next_obs), unit_mask, envs)
                actions[step] = action.T
                log_probs[step] = log_prob
                next_obs, rs, done, infos = envs.step(action.T)
            rewards[step], next_done = rs, done

        states = states.tostring()
        actions = actions.tostring()
        log_probs = log_probs.tostring()
        masks = masks.tostring()
        values = np.around(values, 3).tostring()
        rewards = rewards.tostring()
        ds = ds.tostring()
        version = 0
