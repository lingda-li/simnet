#!/bin/sh

#module load anaconda2
#source activate topaz

CUDA_VISIBLE_DEVICES=10 python ../ml/train-from-one-file.py --test  --data-name totalall.npz --stats-name stats/statsall.npz
