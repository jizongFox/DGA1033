#!/usr/bin/env bash

python train.py --device=cuda  --dataroot=cardiac --method=admm_gc_size --data_aug=False --arch=enet --max_epoch=2 --save_dir=results/compare_loss/gcsize_enet_0.2_positive --eps=0.2 --loss=partial_ce &
python train.py --device=cuda  --dataroot=cardiac --method=admm_gc_size --data_aug=False --arch=enet --max_epoch=2 --save_dir=results/compare_loss/gcsize_enet_0.2_both --eps=0.2 --loss=neg_partial_ce