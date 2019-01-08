#!/usr/bin/env bash
python Inference.py --checkpoint=archive/results/cardiac/FS_enet/best.pth \
--arch=enet \
--dataset=cardiac \
--zoomin --save
python Inference.py --checkpoint=archive/results/cardiac/FS_enet_Daug/best.pth \
--arch=enet \
--dataset=cardiac \
--zoomin --save

python Inference.py --checkpoint=archive/results/cardiac/size_enet_0.0/best.pth \
--arch=enet \
--dataset=cardiac \
--zoomin --save

python Inference.py --checkpoint=archive/results/cardiac/size_enet_0.1/best.pth \
--arch=enet \
--dataset=cardiac \
--zoomin --save

python Inference.py --checkpoint=archive/results/cardiac/size_enet_0.2/best.pth \
--arch=enet \
--dataset=cardiac \
--zoomin --save

python Inference.py --checkpoint=archive/results/cardiac/size_enet_0.4/best.pth \
--arch=enet \
--dataset=cardiac \
--zoomin --save