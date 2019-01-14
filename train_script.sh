#!/usr/bin/env bash
train_main=$1
if [[ $train_main != "train" && $train_main != "train_in" ]];
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

dataequ=$3
if [[ $dataequ != "True" && $dataequ != "prostate" ]];
then
    echo "Avaliable datasets cardiac and prostate, given {$dataset}"
    exit 1
fi


dataaug=$4
if [[ $dataaug != "True" && $dataaug != "False" ]];
then
    echo "Avaliable dataAug options are True or False, given {$dataaug}"
    exit 1
fi
max_epoch=$5
if [[ $max_epoch -gt 200 ]];
then
    echo "max_epoch should be less than 200 , given {$max_epoch}"
    exit 1
fi
size_run=$6
if [[ $size_run != "True" && $size_run != "False" ]];
then
    echo "Avaliable size options are True or False, given {$size_run}"
    exit 1
fi
gc_size_run=$7
if [[ $gc_size_run != "True" && $gc_size_run != "False" ]];
then
    echo "Avaliable gc_size options are True or False, given {$gc_size_run}"
    exit 1
fi


echo "train_main:" $train_main
echo "dataset:" $dataset
echo "dataequ:" $dataequ
echo "dataaug:" $dataaug
echo "max_epoch:"$max_epoch


if [[ $train_main == "train" && $size_run == "True" ]];
then
echo
    python $train_main".py" --device=cuda  --dataroot=$dataset --data_equ=$dataequ --method=fullysupervised --data_aug=$dataaug --arch=cnet --loss=cross_entropy --max_epoch=$max_epoch --save_dir=results/$dataset/FS_cnet_Daug_$dataaug --num_admm_innerloop=1
fi

if [[ $train_main == "train_in" ]];
then
    method_postfix="_in"

fi
echo "method_postfix" $method_postfix




### with only size constraints:
if [[ $size_run == "True"  ]];
then

python $train_main".py" --device=cuda  --dataroot=$dataset --data_equ=$dataequ --method=admm_size$method_postfix --data_aug=$dataaug --arch=cnet  --max_epoch=$max_epoch --save_dir=results/$dataset/$train_main"_size_cnet_Daug_"$dataaug"_0.0" --eps=0.0 &

python $train_main".py" --device=cuda  --dataroot=$dataset --data_equ=$dataequ --method=admm_size$method_postfix --data_aug=$dataaug --arch=cnet --max_epoch=$max_epoch --save_dir=results/$dataset/$train_main"_size_cnet_Daug_"$dataaug"_0.1"  --eps=0.1 &

python $train_main".py" --device=cuda  --dataroot=$dataset --data_equ=$dataequ --method=admm_size$method_postfix --data_aug=$dataaug --arch=cnet --max_epoch=$max_epoch --save_dir=results/$dataset/$train_main"_size_cnet_Daug_"$dataaug"_0.2" --eps=0.2 &

python $train_main".py" --device=cuda  --dataroot=$dataset --data_equ=$dataequ --method=admm_size$method_postfix --data_aug=$dataaug --arch=cnet --max_epoch=$max_epoch --save_dir=results/$dataset/$train_main"_size_cnet_Daug_"$dataaug"_0.4" --eps=0.4
else
echo "size is not estimated."

fi


## with both
if [[ $gc_size_run == "True" ]];
then

python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_gc_size$method_postfix --data_aug=$dataaug --arch=cnet --max_epoch=$max_epoch --save_dir=results/$dataset/$train_main"_gcsize_cnet_Daug_"$dataaug"_0.0" --eps=0.0 &

python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_gc_size$method_postfix --data_aug=$dataaug --arch=cnet --max_epoch=$max_epoch --save_dir=results/$dataset/$train_main"_gcsize_cnet_Daug_"$dataaug"_0.1"  --eps=0.1 &

python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_gc_size$method_postfix --data_aug=$dataaug --arch=cnet --max_epoch=$max_epoch --save_dir=results/$dataset/$train_main"_gcsize_cnet_Daug_"$dataaug"_0.2" --eps=0.2 &

python $train_main".py" --device=cuda  --dataroot=$dataset --method=admm_gc_size$method_postfix --data_aug=$dataaug --arch=cnet --max_epoch=$max_epoch --save_dir=results/$dataset/$train_main"_gcsize_cnet_Daug_"$dataaug"_0.4" --eps=0.4

else
echo "gc_size is not estimated."
fi
