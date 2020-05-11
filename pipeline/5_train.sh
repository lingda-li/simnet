#!/bin/sh

#module load anaconda2
#source activate topaz

rm ../data_spec/models/*.pt

cd ../data_spec
CUDA_VISIBLE_DEVICES=9 python ../ml/train-from-one-file.py  --save --epochs 400000 
