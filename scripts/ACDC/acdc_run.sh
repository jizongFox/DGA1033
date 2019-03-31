#!/usr/bin/env bash
gpu_num=$1
eps=$2
max_epoch=1
use_data_aug=False
save_dir=runs/ACDC
use_tqdm=True

echo 'Parameters:'
echo "GPU:${gpu_num}"
echo "eps for bounds:${eps}"

cd ../
source utils.sh

# now you have the `wait_script`
cd ../
## FS
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Trainer.save_dir="${save_dir}/fs" \
Dataset.dataset_name=cardiac \
Dataset.use_data_aug=$use_data_aug \
ADMM_Method.name=fs \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}

## size
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Trainer.save_dir="${save_dir}/size" \
Dataset.dataset_name=cardiac \
Dataset.use_data_aug=$use_data_aug \
Dataset.metainfoGenerator_dict.eps=${eps} \
ADMM_Method.name=size \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}
## GC_size
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Trainer.save_dir="${save_dir}/gc_size" \
Dataset.dataset_name=cardiac \
Dataset.use_data_aug=$use_data_aug \
Dataset.metainfoGenerator_dict.eps=${eps} \
ADMM_Method.name=gc_size \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}

python admm_research/postprocessing/plot.py --folders "${save_dir}/fs" "${save_dir}/size" "${save_dir}/gc_size"





