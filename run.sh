#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=1 bash train_script.sh train cardiac True 80 False True &
#CUDA_VISIBLE_DEVICES=2 bash train_script.sh train_in cardiac True 80 True False &
#CUDA_VISIBLE_DEVICES=2 bash train_script.sh train_in cardiac True 80 False True
                                                         ## equ, aug epoch size, gc_size
CUDA_VISIBLE_DEVICES=1 bash train_script.sh train prostate True False 80 True False &
CUDA_VISIBLE_DEVICES=2 bash train_script.sh train prostate True False 80 False True
#CUDA_VISIBLE_DEVICES=2 bash train_script.sh train_in prostate True 1 True False
#CUDA_VISIBLE_DEVICES=2 bash train_script.sh train_in prostate True 1 False True