#!/usr/bin/env bash
python train.py --dataroot=cardiac --method=fullysupervised --data_aug=True --arch=enet --loss=cross_entropy --max_epoch=1 --save_dir=results/cardiac/FS_enet_Daug --num_admm_innerloop=1
python train.py --dataroot=cardiac --method=fullysupervised --data_aug=False --arch=enet --loss=cross_entropy --max_epoch=1 --save_dir=results/cardiac/FS_enet --num_admm_innerloop=1

## with only size constraints:

python train.py --dataroot=cardiac --method=admm_size --data_aug=True --arch=enet  --max_epoch=1 --save_dir=results/cardiac/size_enet_Daug_0.0 --eps=0.0
python train.py --dataroot=cardiac --method=admm_size --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/size_enet_0.0 --eps=0.0

python train.py --dataroot=cardiac --method=admm_size --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/size_enet_Daug_0.1  --eps=0.1
python train.py --dataroot=cardiac --method=admm_size --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/size_enet_0.4 --eps=0.1

python train.py --dataroot=cardiac --method=admm_size --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/size_enet_Daug_0.2 --eps=0.2
python train.py --dataroot=cardiac --method=admm_size --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/size_enet_0.4 --eps=0.2

python train.py --dataroot=cardiac --method=admm_size --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/size_enet_Daug_0.4 --eps=0.4
python train.py --dataroot=cardiac --method=admm_size --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/size_enet_0.4 --eps=0.4




## with both 
python train.py --dataroot=cardiac --method=admm_gc_size --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsize_enet_Daug_0.0 --eps=0.0
python train.py --dataroot=cardiac --method=admm_gc_size --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsize_enet_0.0 --eps=0.0

python train.py --dataroot=cardiac --method=admm_gc_size --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsize_enet_Daug_0.1  --eps=0.1
python train.py --dataroot=cardiac --method=admm_gc_size --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsize_enet_0.1 --eps=0.1

python train.py --dataroot=cardiac --method=admm_gc_size --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsize_enet_Daug_0.2 --eps=0.2
python train.py --dataroot=cardiac --method=admm_gc_size --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsize_enet_0.2 --eps=0.2

python train.py --dataroot=cardiac --method=admm_gc_size --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsize_enet_Daug_0.4 --eps=0.4
python train.py --dataroot=cardiac --method=admm_gc_size --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsize_enet_0.4 --eps=0.4


# inequality
## with only size constraints:
python train_in.py --dataroot=cardiac --method=admm_size_in --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/sizeIN_enet_Daug_0.0 --eps=0.0
python train_in.py --dataroot=cardiac --method=admm_size_in --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/sizeIN_enet_0.0 --eps=0.0

python train_in.py --dataroot=cardiac --method=admm_size_in --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/sizeIN_enet_Daug_0.1 --eps=0.1
python train_in.py --dataroot=cardiac --method=admm_size_in --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/sizeIN_enet_0.1 --eps=0.1

python train_in.py --dataroot=cardiac --method=admm_size_in --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/sizeIN_enet_Daug_0.2 --eps=0.2
python train_in.py --dataroot=cardiac --method=admm_size_in --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/sizeIN_enet_0.2 --eps=0.2

python train_in.py --dataroot=cardiac --method=admm_size_in --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/sizeIN_enet_Daug_0.4 --eps=0.4
python train_in.py --dataroot=cardiac --method=admm_size_in --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/sizeIN_enet_0.4 --eps=0.4

# with both
python train_in.py --dataroot=cardiac --method=admm_gc_size_in --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsizeIN_enet_Daug_0.0 --eps=0.0
python train_in.py --dataroot=cardiac --method=admm_gc_size_in --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsizeIN_enet_0.0 --eps=0.0

python train_in.py --dataroot=cardiac --method=admm_gc_size_in --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsizeIN_enet_Daug_0.1--eps=0.1
python train_in.py --dataroot=cardiac --method=admm_gc_size_in --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsizeIN_enet_0.1 --eps=0.1

python train_in.py --dataroot=cardiac --method=admm_gc_size_in --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsizeIN_enet_Daug_0.2 --eps=0.2
python train_in.py --dataroot=cardiac --method=admm_gc_size_in --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsizeIN_enet_0.2 --eps=0.2

python train_in.py --dataroot=cardiac --method=admm_gc_size_in --data_aug=True --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsizeIN_enet_Daug_0.4 --eps=0.4
python train_in.py --dataroot=cardiac --method=admm_gc_size_in --data_aug=False --arch=enet --max_epoch=1 --save_dir=results/cardiac/gcsizeIN_enet_0.4 --eps=0.4