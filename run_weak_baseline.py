import os, numpy as np

networks = ['enet']
epses = [0.01, 0.1, 0.2, 0.4, 0.6]
baselines = ['ADMM_weak', 'ADMM_weak_size']
choice_of_GPU = [0, 1]
lamdas = [0, 0.1, 1, 10]
episodes = [1, 2, 3, 4, 5]
cmds = []
comment = 'with_eval'

class GPU_number_iter(object):
    def __init__(self,choices) -> None:
        super().__init__()
        assert type(choices)==list
        self.choices = choices
        self.iter_choice = iter(choices)
    def __call__(self):
        try:
            return self.iter_choice.__next__()
        except:
            self.iter_choice = iter(self.choices)
            return self.iter_choice.__next__()
gpu_n = GPU_number_iter(choice_of_GPU)



for net in networks:
    for baseline in baselines:
        for lamda in lamdas:
            for eps in epses:
                for episode in episodes:
                    cmd = "MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=%d  python weakly-main.py --netarch %s --baseline %s --lamda %f --eps %f --episode %d --comments %s" % (
                        gpu_n(),
                        net,
                        baseline,
                        lamda,
                        eps,
                        episode,
                        comment)
                    cmds.append(cmd)
for net in networks:
    for lamda in lamdas:
        for episode in episodes:
            cmd =  "MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=%d  python weakly-main.py --netarch %s --baseline %s --lamda %f --episode %d --comments %s" % (
                gpu_n(),net,'ADMM_weak_gc',lamda, episode,comment)
            cmds.append(cmd)
print(cmds.__len__())


if __name__ == '__main__':
    from multiprocessing import Pool

    p = Pool(18)
    p.map(os.system, cmds)
