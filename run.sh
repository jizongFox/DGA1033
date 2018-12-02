#!/usr/bin/env bash
rm -rf ./runs/*
bash ./run_compare_negative.sh &
bash ./run_cardiac_vs_eps.sh &
bash ./run_prostate_vs_eps.sh
