#!/usr/bin/env bash



mainfolder=parameterSearch/log_equalize/prostate/WeaklyAnnotations/

subfolder=" "


for name in \
name_prostate_kernal_size_5_lamda_0.01_sigma_0.001_dilation_level_6 \
name_prostate_kernal_size_5_lamda_0.1_sigma_0.001_dilation_level_6 \
name_prostate_kernal_size_5_lamda_1.0_sigma_0.001_dilation_level_6 \
name_prostate_kernal_size_5_lamda_0.01_sigma_0.001_dilation_level_7 \
name_prostate_kernal_size_5_lamda_0.1_sigma_0.001_dilation_level_7 \
name_prostate_kernal_size_5_lamda_1.0_sigma_0.001_dilation_level_7 \
; do
subfolder=$subfolder' '$mainfolder$name

echo $subfolder
done


python3.6 viewer.py \
admm_research/dataset/PROSTATE/train/GT \
admm_research/dataset/PROSTATE/train/centroid \
$subfolder \
--img_source=admm_research/dataset/PROSTATE/train/Img \





