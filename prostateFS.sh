#!/usr/bin/env bash
python train.py --device=cuda --dataroot=prostate --data_equ=False --method=fullysupervised --data_aug=False --arch=cnet --loss=cross_entropy --max_epoch=50  \
--save_dir=results/prostate/FS_test/FS_cnet_baseline

#
#python report.py \
#--postfix test \
#--folders \
# archive/results/prostate/FS_test/FS_enet_Daug_True_equ_False\
# archive/results/prostate/FS_test/FS_enet_Daug_True_equ_True \
#--file \
#metrics.csv \
#--axis \
#2 \
#3 \
#--y_lim \
#0.6 \
#1 \
#
#
