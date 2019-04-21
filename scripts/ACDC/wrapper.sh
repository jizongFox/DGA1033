#!/usr/bin/env bash
wrapper(){
    commend=$1
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
     --account=def-mpederso \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
     acdc_3d_run.sh 0 $commend
}
wrapper run_size 72
wrapper run_fs 24
wrapper run_soft 24
wrapper run_gc 72
wrapper run_gc_size 72
