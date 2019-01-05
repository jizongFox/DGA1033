#!/usr/bin/env bash

python parameterSearch.py --name cardiac --output_dir parameterSearch
python parameterSearch.py --name prostate --output_dir parameterSearch  --folder_name=centroid