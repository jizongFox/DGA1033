#!/usr/bin/env bash
wrapper(){
    commend=$1
    class=$2
    eps=$3
    hour=$4
    #echo "Running: ${model_num}"
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    sbatch  --job-name="${commend}_${class}_${eps}" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=16000M \
     --time=0-${hour}:00 \
     --account=def-mpederso \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
     acdc_3d_run.sh 0 $class $eps $commend
}

# baseline
wrapper run_fs LV 0 1
wrapper run_gc LV 0 1

# 0.1 LV
wrapper run_soft LV 0.1 1
wrapper run_size LV 0.1 1
wrapper run_gc_size LV 0.1 1

# 0.2 LV
wrapper run_soft LV 0.2 1
wrapper run_size LV 0.2 1
wrapper run_gc_size LV 0.2 1

# 0.4 LV
wrapper run_soft LV 0.4 1
wrapper run_size LV 0.4 1
wrapper run_gc_size LV 0.4 1

# 0.6 LV
wrapper run_soft LV 0.6 1
wrapper run_size LV 0.6 1
wrapper run_gc_size LV 0.6 1



# baseline
wrapper run_fs RV 0 1
wrapper run_gc RV 0 1

# 0.1 RV
wrapper run_soft RV 0.1 1
wrapper run_size RV 0.1 1
wrapper run_gc_size RV 0.1 1

# 0.2 RV
wrapper run_soft RV 0.2 1
wrapper run_size RV 0.2 1
wrapper run_gc_size RV 0.2 1

# 0.4 RV
wrapper run_soft RV 0.4 1
wrapper run_size RV 0.4 1
wrapper run_gc_size RV 0.4 1

# 0.6 RV
wrapper run_soft RV 0.6 1
wrapper run_size RV 0.6 1
wrapper run_gc_size RV 0.6 1