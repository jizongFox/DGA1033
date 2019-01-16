#!/usr/bin/env bash
#python train.py --device=cuda --dataroot=prostate --data_equ=False --method=fullysupervised --data_aug=True --arch=cnet --loss=cross_entropy --max_epoch=2  --num_admm_innerloop=1 --vis_during_training --eps=0.1 --optim_inner_loop_num=1 \
#--save_dir=results/prostate/FS_test/FS_cnet_Daug_True_equ_False
#python train.py --device=cuda --dataroot=prostate --data_equ=True --method=fullysupervised --data_aug=True --arch=cnet --loss=cross_entropy --max_epoch=2  --num_admm_innerloop=1 --vis_during_training --eps=0.1 --optim_inner_loop_num=1 \
#--save_dir=results/prostate/FS_test/FS_cnet_Daug_True_equ_True

python train.py --device=cuda --dataroot=prostate --data_equ=False --method=fullysupervised --data_aug=True --arch=enet --loss=cross_entropy --max_epoch=2  --num_admm_innerloop=1 --vis_during_training --eps=0.1 --optim_inner_loop_num=1 \
--save_dir=results/prostate/FS_test/FS_enet_Daug_True_equ_False
python train.py --device=cuda --dataroot=prostate --data_equ=True --method=fullysupervised --data_aug=True --arch=cnet --loss=cross_entropy --max_epoch=2  --num_admm_innerloop=1 --vis_during_training --eps=0.1 --optim_inner_loop_num=1 \
--save_dir=results/prostate/FS_test/FS_enet_Daug_True_equ_True

python report.py \
--postfix test \
--folders \
archive/results/prostate/FS_test/FS_enet_Daug_True_equ_False\
archive/results/prostate/FS_test/FS_enet_Daug_True_equ_True \
--file \
metrics.csv \
--axis \
2 \
3 \
--y_lim \
0.6 \
1 \


