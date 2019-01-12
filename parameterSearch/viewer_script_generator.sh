#!/usr/bin/env bash
input=$1
echo $input
inputfolder=$2
echo $inputfolder
showalllamda=$3
echo $showalllamda
inputzip=$input'.zip'
echo 'inputzip: '$inputzip
#

if [ ! -d "parameterSearch/$input" ]; then
  # Control will enter here if $DIRECTORY exists.
  echo ">>> unzip from $inputzip to $input "
  unzip -x -q  parameterSearch/$inputzip -d parameterSearch/
fi


RES=$(python parameterSearch/viewer_wraper.py --csv_path=parameterSearch/$input/prostate/$inputfolder/prostate.csv \
--img_source=admm_research/dataset/PROSTATE/train/Img \
$showalllamda)
echo $RES

eval $RES