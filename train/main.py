import argparse
from cmath import exp
import master
import worker
import torch.multiprocessing as mp
import time
import torch
from agent_model import MobaAgent, MobaNet


if __name__ == '__main__':

    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    parser.add_argument('--current-version', default=0, help='当前的版本')
    parser.add_argument('--num-worker', default=2, help='worker的数量')
    parser.add_argument('--observation-length', default=1000)
    parser.add_argument('--action-length', default=100)
    parser.add_argument('--learning-rate', default=2.5e-4, help='the learning rate of the optimizer')
    parser.add_argument('--batch-size', default=512)
    parser.add_argument('--action-space', default=[100, 6, 4, 4, 4, 4, 7, 49])
    args = parser.parse_args()

    moba_net = MobaNet().share_memory()
    moba_master = master.Master(moba_net, num_worker=args.num_worker, version=0)
    workers = []
    processes = []

    for i in range(args.num_worker):
        moba_agent = MobaAgent(moba_master.shared_model, args.action_space)
        moba_worker = worker.Worker(
            agent=moba_agent,
            batch_size=1024,
            optimizer=torch.optim.Adam(params=moba_master.shared_model.parameters(), lr=2.5e-3),
            use_gpu=False,
            identity=i,
            version=moba_master.version)
        workers.append(moba_worker)
        moba_worker.run_worker()
        
    while True:
        """
        1 master.get_replays_from_redis()
        2 dispatch exp to worker 
        3 get grad from worker 
        4 update net parameters (share_memory)
        5 send model parameter to redis 
        """
        moba_master.get_replays_from_redis()
        k = 10
        for n in range(k):
            for m_worker in workers:
                try:
                    replay = moba_master.train_queue.get(block=False)
                    if replay is not None:
                        m_worker.put_replays(replay)
                    else:
                        time.sleep(0)
                except:
                    continue
        time.sleep(0)







