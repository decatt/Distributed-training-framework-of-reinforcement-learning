import torch
import torch.multiprocessing as mp
from agent_model import MobaNet
import numpy as np
import time

"""class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()"""


class Master:
    def __init__(self, net, num_worker, version, queue_length=10000):
        self.train_queue = mp.Queue(queue_length)
        self.grad_queue = mp.Queue(queue_length)
        self.version = version
        self.num_worker = num_worker
        self.shared_model = net
        self.state_shape = (10,10,27)
        self.action_length = 178
        self.action_shape = [100, 6, 4, 4, 4, 4, 7, 49]
        self.batch_size = 1024
        self.num_mdps = 10

    def get_replays_from_redis(self):
        gamma = 0.99
        gae_lambda = 0.95
        update_epochs = 1
        mini_batch_size = 128

        # get following from redis
        states = np.ones((self.batch_size, self.num_mdps, )+self.state_shape)
        actions = np.ones((self.batch_size, self.num_mdps, len(self.action_shape)))
        rewards = np.ones((self.batch_size, self.num_mdps))
        values = np.ones((self.batch_size, self.num_mdps))
        ds = np.ones((self.batch_size, self.num_mdps))
        masks = np.ones((self.batch_size, self.num_mdps, self.action_length))
        log_probs = np.ones((self.batch_size, self.num_mdps))
        version = 0

        advantages = np.zeros((self.batch_size, self.num_mdps))

        last_gae_lam = 0
        for t in reversed(range(self.batch_size-1)):
            next_non_terminal = 1.0 - ds[t + 1]
            next_values = values[t + 1]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        states = states.reshape((self.batch_size * self.num_mdps,)+self.state_shape)
        actions = actions.reshape((-1, len(self.action_shape)))
        values = values.reshape((self.batch_size * self.num_mdps,))
        masks = masks.reshape((-1, self.action_length))
        log_probs = log_probs.reshape((self.batch_size * self.num_mdps,))
        advantages = advantages.reshape((self.batch_size * self.num_mdps,))

        inds = np.arange(self.batch_size * self.num_mdps, )
        for i_epoch_pi in range(update_epochs):
            np.random.shuffle(inds)
            for start in range(0, self.batch_size * self.num_mdps, mini_batch_size):
                end = start + mini_batch_size
                mini_batch_ind = inds[start:end]
                b_states = states[mini_batch_ind]
                b_actions = actions[mini_batch_ind]
                b_values = values[mini_batch_ind]
                b_masks = masks[mini_batch_ind]
                b_log_probs = log_probs[mini_batch_ind]
                b_advantages = advantages[mini_batch_ind]
                replay = (b_states, b_actions, b_values, b_masks, b_log_probs, version, b_advantages)
                self.train_queue.put(replay)

    def export_onnx(self):
        """

        :return:
        """

    """def run(self):
        p = mp.Process(target=self.get_replays_from_redis())
        print('star process' + str(p.pid))
        self.process_id = p.pid
        p.start()
        return p.pid

    # input grad: (version, gard)
    def get_grad(self, grads):
        print('get grad' + str(self.process_id))
        try:
            self.grad_queue.put(grads[1])
        except:
            print('empty grad queue')

    def backward(self):
        print('backward' + str(self.process_id))
        grads = self.grad_queue.get()
        for grad, shared_param in zip(grads, self.shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = grad
        self.shared_optimizer.step()"""
