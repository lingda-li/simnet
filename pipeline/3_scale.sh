#!/bin/sh

#module load anaconda2
#source activate topaz

python ../dp/3_scale.py --save $1.mlu.t*.npz
