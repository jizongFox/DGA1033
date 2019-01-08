#!/usr/bin/env bash

python report.py \
--postfix sizeonly \
--folders \
archive/results/cardiac/size_enet_0.0 \
archive/results/cardiac/size_enet_0.1 \
archive/results/cardiac/size_enet_0.2 \
archive/results/cardiac/size_enet_0.4 \
archive/results/cardiac/FS_enet \
--file \
metrics.csv \
--axis \
1 \
2 \
3 \
--y_lim \
0.6 \
1 \
#--interpolate

python report.py \
--postfix gcsize \
--folders \
archive/results/cardiac/gcsize_enet_0.0 \
archive/results/cardiac/gcsize_enet_0.1 \
archive/results/cardiac/gcsize_enet_0.2 \
archive/results/cardiac/gcsize_enet_0.4 \
archive/results/cardiac/FS_enet \
--file \
metrics.csv \
--axis \
1 \
2 \
3 \
--y_lim \
0.6 \
1 \
#--interpolate

## inequality
python report.py \
--postfix INsizeonly \
--folders \
archive/results/cardiac/sizeIN_enet_0.0 \
archive/results/cardiac/sizeIN_enet_0.1 \
archive/results/cardiac/sizeIN_enet_0.2 \
archive/results/cardiac/sizeIN_enet_0.4 \
archive/results/cardiac/FS_enet \
--file \
metrics.csv \
--axis \
1 \
2 \
3 \
--y_lim \
0.6 \
1 \
#--interpolate

python report.py \
--postfix INgcsize \
--folders \
archive/results/cardiac/gcsizeIN_enet_0.0 \
archive/results/cardiac/gcsizeIN_enet_0.1 \
archive/results/cardiac/gcsizeIN_enet_0.2 \
archive/results/cardiac/gcsizeIN_enet_0.4 \
archive/results/cardiac/FS_enet \
--file \
metrics.csv \
--axis \
1 \
2 \
3 \
--y_lim \
0.6 \
1 \
#--interpolate

