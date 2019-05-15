#!/usr/bin/env bash
#!/usr/bin/env bash
cd ..
cd ..
wrapper(){
    hour=$1
    command=$2
    #echo "Running: ${model_num}"
    module load python/3.6
    source $HOME/torchenv36/bin/activate
    module load scipy-stack
    echo $command > tmp.sh
    sed -i '1i\#!/bin/bash' tmp.sh
    sbatch  --job-name="${commend}" \
     --nodes=1  \
     --gres=gpu:1 \
     --cpus-per-task=6  \
     --mem=12000M \
     --time=0-${hour}:00 \
     --account=def-mpederso \
     --mail-user=jizong.peng.1@etsmtl.net \
     --mail-type=ALL   \
    ./tmp.sh
}
#cd ..
time=1
# as in the paper
wrapper $time  "python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/gc_size"
# single head
wrapper $time  "python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/gc ADMM_Method.p_v=0.0"
# single head with 1 subhead
wrapper $time  "python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/size ADMM_Method.p_u=0.0"
# two head with 1 subhead
wrapper $time  "python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/soft3d ADMM_Method.name=soft3d"






#cd ..
#cd ..
#python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/gc_size
#python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/gc ADMM_Method.p_v=0.0
#python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/size ADMM_Method.p_u=0.0
#python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/soft3d ADMM_Method.name=soft3d
