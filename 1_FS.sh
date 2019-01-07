#!/usr/bin/env bash
train_main=$1
if [[ $train_main != "train" && $dataset != "train_in" ]];
then
    echo "Avaliable scripts are train and train_in, given {$train_main}"
    exit 1
fi

dataset=$2
if [[ $dataset != "cardiac" && $dataset != "prostate" ]];
then
    echo "Avaliable datasets cardiac and prostate, given {$dataset}"
    exit 1
fi
dataaug=$3
if [[ $dataaug != "True" && $dataaug != "False" ]];
then
    echo "Avaliable dataAug options are True or False, given {$dataaug}"
    exit 1
fi
max_epoch=$4
if [[ $max_epoch -gt 200 ]];
then
    echo "max_epoch should be less than 200 , given {$max_epoch}"
    exit 1
fi


echo "train_main:" $train_main
echo "dataset:" $dataset
echo "dataaug:" $dataaug
echo "max_epoch:"$max_epoch

if [[ $train_main == "train" ]];
then

    python $train_main".py" --device=cuda  --dataroot=$dataset --method=fullysupervised --data_aug=$dataaug --arch=enet --loss=cross_entropy --max_epoch=$max_epoch --save_dir=results/cardiac/FS_enet_Daug_$dataaug --num_admm_innerloop=1
fi

### with only size constraints:

#echo $train_main"_size_enet_Daug_"$dataaug"_0.0"
python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_size --data_aug=$dataaug --arch=enet  --max_epoch=$max_epoch --save_dir=results/cardiac/$train_main"_size_enet_Daug_"$dataaug"_0.0" --eps=0.0 &

python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_size --data_aug=$dataaug --arch=enet --max_epoch=$max_epoch --save_dir=results/cardiac/$train_main"_size_enet_Daug_"$dataaug"_0.1"  --eps=0.1  &

python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_size --data_aug=$dataaug --arch=enet --max_epoch=$max_epoch --save_dir=results/cardiac/$train_main"_size_enet_Daug_"$dataaug"_0.2" --eps=0.2 &

python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_size --data_aug=$dataaug --arch=enet --max_epoch=$max_epoch --save_dir=results/cardiac/$train_main"_size_enet_Daug_"$dataaug"_0.4" --eps=0.4


## with both
python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_gc_size --data_aug=$dataaug --arch=enet --max_epoch=$max_epoch --save_dir=results/cardiac/$train_main"_gcsize_enet_Daug_"$dataaug"_0.0" --eps=0.0 &

python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_gc_size --data_aug=$dataaug --arch=enet --max_epoch=$max_epoch --save_dir=results/cardiac/$train_main"_gcsize_enet_Daug_"$dataaug"_0.1"  --eps=0.1 &

python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_gc_size --data_aug=$dataaug --arch=enet --max_epoch=$max_epoch --save_dir=results/cardiac/$train_main"_gcsize_enet_Daug_"$dataaug"_0.2" --eps=0.2 &

python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_gc_size --data_aug=$dataaug --arch=enet --max_epoch=$max_epoch --save_dir=results/cardiac/$train_main"_gcsize_enet_Daug_"$dataaug"_0.4" --eps=0.4


