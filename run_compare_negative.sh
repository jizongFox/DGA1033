#!/usr/bin/env bash
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python train.py --device=cuda --dataroot=cardiac --optim_inner_loop_num=3 --stop_dilation_epoch=200  --ignore_negative=False --eps=0.01 &

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python train.py --device=cuda --dataroot=cardiac --optim_inner_loop_num=3 --stop_dilation_epoch=200 --ignore_negative=True --eps=0.01 &

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 python train.py --device=cuda --dataroot=prostate --optim_inner_loop_num=3 --stop_dilation_epoch=100  --ignore_negative=False --eps=0.01 &

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=3 python train.py --device=cuda --dataroot=prostate --optim_inner_loop_num=3 --stop_dilation_epoch=100 --ignore_negative=True --eps=0.01