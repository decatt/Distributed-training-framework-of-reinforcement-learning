import time
import torch
import torch.multiprocessing as mp
import agent_model
from tensorboardX import SummaryWriter


class Worker:
    def __init__(self, agent: agent_model.MobaAgent, batch_size: int, optimizer, use_gpu: bool, identity: int,
                 version: int, queue_length=10000):
        self.process_id = identity
        self.version = version
        self.agent = agent
        self.batch_size = batch_size
        self.train_queue = mp.Queue(queue_length)
        self.grad_queue = mp.Queue(queue_length)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.optimizer = optimizer
        self.process = None

    def put_replays(self, replays):
        self.train_queue.put(replays)
        # print('put replay ' + str(self.process_id))

    def run_worker(self):
        self.process = mp.Process(target=self.process_function)
        self.process.start()
        print('star process' + str(self.process.pid))

    def process_function(self):
        """grad_queue.put(grad)"""
        clip_coef = 0.1
        # gamma = 0.998
        # gae_lambda = 0.95
        ent_coef = 0.05
        vf_coef = 0.5
        clip_v_loss = False
        writer = SummaryWriter('./Records')
        update = 0
        path_pt = 'moba_net.pt'
        num_updates = 1000
        while update < num_updates:
            try:
                replay = self.train_queue.get(block=False)
                if replay is None:
                    time.sleep(0)
                    continue
                else:
                    frac = 1.0 - (update - 1.0) / num_updates
                    lr_now = 2.5e-3 * frac
                    self.optimizer.param_groups[0]['lr'] = lr_now
                    update = update + 1
                    states, actions, values, masks, log_probs, version, advantages = replay
                    states = torch.Tensor(states)
                    actions = torch.Tensor(actions)
                    # rewards = torch.Tensor(rewards)
                    values = torch.Tensor(values)
                    # ds = torch.Tensor(ds)
                    masks = torch.Tensor(masks)
                    log_probs = torch.Tensor(log_probs)
                    advantages = torch.Tensor(advantages)
                    returns = values + advantages
                    b_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    new_log_prob, entropy, new_values = self.agent.get_log_prob_entropy_value(states, action=actions.T,
                                                                                              masks=masks)
                    new_log_prob = new_log_prob.to(self.device)
                    entropy = entropy.to(self.device)
                    ratio = (new_log_prob - log_probs).exp()
                    # Policy loss
                    pg_loss1 = -b_advantages * ratio
                    pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy.mean()
                    # Value loss
                    if clip_v_loss:
                        v_loss_un_clipped = ((new_values - returns) ** 2)
                        v_clipped = values + torch.clamp(new_values - values, -clip_coef, clip_coef)
                        v_loss_clipped = (v_clipped - returns) ** 2
                        v_loss_max = torch.max(v_loss_un_clipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_values - returns) ** 2).mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
                    writer.add_scalar('loss', loss, update)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                    # self.grad_queue.put((self.version, self.agent.get_grads()))
                    self.optimizer.step()
                    print('updated' + str(self.process_id))
                    if update % 10 == 0:
                        torch.save(self.agent.net.state_dict(), path_pt)
                        # torch.onnx.export(self.agent.net, torch.zeros((1, 10, 10, 27)), "moba_net.onnx")
            except:
                continue
        writer.close()
        print('process end')

    def get_gradient(self):
        """:return grad"""
        return self.grad_queue.get()
