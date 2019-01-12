#!/usr/bin/env bash

outputdir='log_equalize'
rm -rf $outputdir
python parameterSearch/parameterSearch.py --name prostate --folder_name=centroid --output_dir=$outputdir \
                          --sigmas 100 10 1 0.1 0.01 0.001 --lambdas 0.0 0.01 0.1 1.0 --kernel_sizes 3 5 \
                          --dilation_levels  5 6 7 9 11 --equalize  --debug


python parameterSearch/parameterSearch.py --name prostate --folder_name=WeaklyAnnotations --output_dir=$outputdir \
                          --sigmas 100 10 1 0.1 0.01 0.001 --lambdas 0.0 0.01 0.1 1.0 --kernel_sizes 3 5 \
                          --dilation_levels  5 6 7  --equalize  --debug

zip -q -r parameterSearch/$outputdir.zip  $outputdir
rm -rf $outputdir

