#!/usr/bin/env bash
gpu_num=$1
commend=$2
max_epoch=300
choosen_class=RV
subfolder="${choosen_class}_prior/"
save_dir=$subfolder
use_tqdm=True
set -e

echo 'Parameters:'
echo "GPU:${gpu_num}"

cd ../
source utils.sh

# now you have the `wait_script`
cd ../
# FS
run_fs(){
rm -rf "runs/${save_dir}/fs"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Trainer.save_dir="runs/${save_dir}/fs" \
Dataset.dataset_name=cardiac \
ADMM_Method.name=fs \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}
rm -rf "archives/${save_dir}/fs"
mv -f "runs/${save_dir}/fs" "archives/${save_dir}"
}

run_soft(){
rm -rf "runs/${save_dir}/soft3d"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Trainer.save_dir="runs/${save_dir}/soft3d" \
Optim.lr=0.0005 \
Dataset.dataset_name=cardiac \
ADMM_Method.name=soft3d \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}
rm -rf "archives/${save_dir}/soft3d"
mv -f "runs/${save_dir}/soft3d" "archives/${save_dir}"
}

## size
run_size(){
rm -rf "runs/${save_dir}/size"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Trainer.save_dir="runs/${save_dir}/size" \
Dataset.dataset_name=cardiac \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm} \
ADMM_Method.p_u=0
rm -rf "archives/${save_dir}/size"
mv -f "runs/${save_dir}/size" "archives/${save_dir}"
}

run_gc(){
rm -rf "runs/${save_dir}/gc"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Trainer.save_dir="runs/${save_dir}/gc" \
Dataset.dataset_name=cardiac \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm} \
ADMM_Method.p_v=0
rm -rf "archives/${save_dir}/gc"
mv -f "runs/${save_dir}/gc" "archives/${save_dir}"
}
# GC_size
run_gc_size(){
rm -rf "runs/${save_dir}/gc_size"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Trainer.save_dir="runs/${save_dir}/gc_size" \
Dataset.dataset_name=cardiac \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}
rm -rf "archives/${save_dir}/gc_size"
mv -f "runs/${save_dir}/gc_size" "archives/${save_dir}"
}
mkdir -p "archives/${save_dir}"
#run_fs
echo $commend
$commend
#run_size
#wait_script
#run_gc_size
#
#python admm_research/postprocessing/plot.py --folders "archives/${save_dir}/fs" "archives/${save_dir}/size" "archives/${save_dir}/gc_size" --file=wholeMeter.csv --classes tra_2d_dice_DSC1 val_2d_dice_DSC1 val_3d_dice_DSC1