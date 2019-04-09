#!/usr/bin/env bash
gpu_num=$1
eps=$2
max_epoch=500
save_dir=PROSTATE_gt_oracle
use_tqdm=False
dataset_name=prostate_aug
use_data_aug=False
subfolder=erosion
set -e

echo 'Parameters:'
echo "GPU:${gpu_num}"
echo "eps for bounds:${eps}"

cd ../
source utils.sh

# now you have the `wait_script`
cd ../
# FS
run_fs(){
rm -rf "runs/${save_dir}/fs"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Arch.name=cnet \
Trainer.save_dir="runs/${save_dir}/fs" \
Dataset.dataset_name=${dataset_name} \
Dataset.use_data_aug=$use_data_aug \
Dataset.subfolder=$subfolder \
ADMM_Method.name=fs \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}
rm -rf "archives/${save_dir}/fs"
mv -f "runs/${save_dir}/fs" "archives/${save_dir}"
}

## size
run_size(){
rm -rf "runs/${save_dir}/size"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Arch.name=cnet \
Trainer.save_dir="runs/${save_dir}/size" \
Dataset.dataset_name=${dataset_name} \
Dataset.use_data_aug=$use_data_aug \
Dataset.subfolder=$subfolder \
Dataset.metainfoGenerator_dict.eps=${eps} \
ADMM_Method.name=size \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}
rm -rf "archives/${save_dir}/size"
mv -f "runs/${save_dir}/size" "archives/${save_dir}"
}
# GC_size
run_gc_size(){
rm -rf "runs/${save_dir}/gc_size"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Arch.name=cnet \
Trainer.save_dir="runs/${save_dir}/gc_size" \
Dataset.dataset_name=${dataset_name} \
Dataset.use_data_aug=$use_data_aug \
Dataset.subfolder=$subfolder \
Dataset.metainfoGenerator_dict.eps=${eps} \
ADMM_Method.name=gc_size \
ADMM_Method.lamda=0.5 \
ADMM_Method.sigma=0.005 \
ADMM_Method.kernel_size=5 \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}
rm -rf "archives/${save_dir}/gc_size"
mv -f "runs/${save_dir}/gc_size" "archives/${save_dir}"
}

mkdir -p "archives/${save_dir}"
#run_fs
#run_size
#wait_script
run_gc_size

#python admm_research/postprocessing/plot.py --folders "archives/${save_dir}/fs" "archives/${save_dir}/size" "archives/${save_dir}/gc_size" --file=wholeMeter.csv --classes tra_2d_dice_DSC1 val_2d_dice_DSC1 val_3d_dice_DSC1

