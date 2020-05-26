#!/bin/sh

#module load anaconda2
#source activate topaz

python ../dp/4_combine.py normall_$1.mlu.t*.npz
