#!/usr/bin/env bash
gpu_num=$1
choosen_class=$2
new_eps=$3
commend=$4

if [ ${choosen_class} != "LV" ] && [ ${choosen_class} != "RV" ];
then
    echo "choosen_class must be LV or RV, given ${choosen_class}."
    exit 1
fi

config_file_path="config/config_3D_${choosen_class}.yaml"

max_epoch=350
gc_use_prior=False
subfolder="${choosen_class}_prior/"
save_dir="${subfolder}eps_search_${new_eps}"
use_tqdm=True
set -e

echo 'Parameters:'
echo "GPU:${gpu_num}"
echo "Choosen config file: ${config_file_path}"
echo "3D size eps: ${new_eps}"
echo "Run command: ${commend}"
cd ../
source utils.sh

# now you have the `wait_script`
cd ../
# FS
run_fs(){
rm -rf "runs/${save_dir}/fs"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Config=$config_file_path \
Trainer.save_dir="runs/${save_dir}/fs" \
Dataset.dataset_name=cardiac \
Dataset.choosen_class=$choosen_class \
Dataset.subfolder=$subfolder \
ADMM_Method.name=fs \
ADMM_Method.gc_use_prior=$gc_use_prior \
ADMM_Method.new_eps=$new_eps \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}
rm -rf "archives/${save_dir}/fs"
mv -f "runs/${save_dir}/fs" "archives/${save_dir}"
}

run_soft(){
rm -rf "runs/${save_dir}/soft3d"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Config=$config_file_path \
Trainer.save_dir="runs/${save_dir}/soft3d" \
Optim.lr=0.0005 \
Dataset.dataset_name=cardiac \
Dataset.choosen_class=$choosen_class \
Dataset.subfolder=$subfolder \
ADMM_Method.name=soft3d \
ADMM_Method.gc_use_prior=$gc_use_prior \
ADMM_Method.new_eps=$new_eps \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm}
rm -rf "archives/${save_dir}/soft3d"
mv -f "runs/${save_dir}/soft3d" "archives/${save_dir}"
}

## size
run_size(){
rm -rf "runs/${save_dir}/size"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Config=$config_file_path \
Trainer.save_dir="runs/${save_dir}/size" \
Dataset.dataset_name=cardiac \
Dataset.choosen_class=$choosen_class \
Dataset.subfolder=$subfolder \
ADMM_Method.new_eps=$new_eps \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm} \
ADMM_Method.p_u=0 \
ADMM_Method.gc_use_prior=$gc_use_prior
rm -rf "archives/${save_dir}/size"
mv -f "runs/${save_dir}/size" "archives/${save_dir}"
}

run_gc(){
rm -rf "runs/${save_dir}/gc"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Config=$config_file_path \
Trainer.save_dir="runs/${save_dir}/gc" \
Dataset.dataset_name=cardiac \
Dataset.choosen_class=$choosen_class \
Dataset.subfolder=$subfolder \
ADMM_Method.new_eps=$new_eps \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm} \
ADMM_Method.p_v=0 \
ADMM_Method.gc_use_prior=$gc_use_prior
rm -rf "archives/${save_dir}/gc"
mv -f "runs/${save_dir}/gc" "archives/${save_dir}"
}
# GC_size
run_gc_size(){
rm -rf "runs/${save_dir}/gc_size"
CUDA_VISIBLE_DEVICES=${gpu_num} python main.py  \
Config=$config_file_path \
Trainer.save_dir="runs/${save_dir}/gc_size" \
Dataset.dataset_name=cardiac \
Dataset.choosen_class=$choosen_class \
Dataset.subfolder=$subfolder \
ADMM_Method.new_eps=$new_eps \
Trainer.max_epoch=${max_epoch} \
Trainer.use_tqdm=${use_tqdm} \
ADMM_Method.gc_use_prior=$gc_use_prior
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