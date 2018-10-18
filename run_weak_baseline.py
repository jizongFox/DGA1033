import os, numpy as np

networks = ['enet', 'unet']
epses = [0.01, 0.1, 0.2, 0.4, 0.6]
baselines = ['ADMM_weak', 'ADMM_weak_size']
choice_of_GPU = [0, 1]
lamdas = [0, 0.1, 1, 10]
episodes = [1, 2, 3, 4, 5]
cmds = []

for net in networks:
    for baseline in baselines:
        for lamda in lamdas:
            for eps in epses:
                for episode in episodes:
                    cmd = "MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=%d  python weakly-main.py --netarch %s --baseline %s --lamda %f --eps %f --episode %d --comments %s" % (
                        np.random.choice(choice_of_GPU),
                        net,
                        baseline,
                        lamda,
                        eps,
                        episode,
                        'official')
                cmds.append(cmd)
for net in networks:
    for lamda in lamdas:
        for episode in episodes:
            cmd =  "MP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=%d  python weakly-main.py --netarch %s --baseline %s --lamda %f --eps %f --episode %d --comments %s" % (
                np.random.choice(choice_of_GPU),net,'ADMM_weak_gc',lamda, 0,episode,'official')
            cmds.append(cmd)
print(cmds.__len__())


if __name__ == '__main__':
    from multiprocessing import Pool

    p = Pool()
    p.map(os.system, cmds)
