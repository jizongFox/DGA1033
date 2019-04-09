#!/usr/bin/env bash
wrapper(){
    max_epoch=$1
    hour=$2
    #echo "Running: ${model_num}"
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    sbatch  --job-name="prostate" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=32000M \
     --time=0-${hour}:00 \
     --account=def-chdesa \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
     prostate_run.sh 0 500 0.1
}