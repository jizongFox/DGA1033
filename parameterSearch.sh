#!/usr/bin/env bash
rm -rf log_equalize
python parameterSearch.py --name prostate --folder_name=centroid --output_dir=log_equalize \
                          --sigmas 100 10 1 0.1 0.01 0.001 --lambdas 0.0 0.01 0.1 1.0 --kernel_sizes 3 5 \
                          --dilation_levels  5 6 7 9 11 --equalize

RES=$(python viewer_wraper.py --csv_path=log_equalize/prostate/centroid/prostate.csv --img_source=admm_research/dataset/PROSTATE/train/Img)

echo $RES -> log_equalize/view_centroid.sh

#eval $RES

python parameterSearch.py --name prostate --folder_name=WeaklyAnnotations --output_dir=log_equalize \
                          --sigmas 100 10 1 0.1 0.01 0.001 --lambdas 0.0 0.01 0.1 1.0 --kernel_sizes 3 5 \
                          --dilation_levels  5 6 7  --equalize

RES=$(python viewer_wraper.py --csv_path=log_equalize/prostate/WeaklyAnnotations/prostate.csv --img_source=admm_research/dataset/PROSTATE/train/Img)

echo $RES -> log_equalize/view_weaklyannotations.sh

zip -q -r log_equalize.zip log_equalize
mv log_equalize.zip parameterSearch/log_equalize.zip

#bash log_equalize/view_centroid.sh