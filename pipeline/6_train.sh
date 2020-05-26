#!/bin/sh

#module load anaconda2
#source activate topaz

CUDA_VISIBLE_DEVICES=9 python ../ml/train-from-one-file.py  --save --epochs 10000 --data-name totalall.npz --stats-name stats/statsall.npz
