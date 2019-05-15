#!/usr/bin/env bash
cd ..
cd ..
python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/gc_size
python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/gc ADMM_Method.p_v=0.0
python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/size ADMM_Method.p_u=0.0
python main.py Config=config/config_Prostate.yaml Trainer.save_dir=runs/prostate_eps_0.1/soft3d ADMM_Method.name=soft3d
