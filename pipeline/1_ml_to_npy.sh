#!/bin/bash

#module load anaconda2
#source activate topaz

rm data/*.npz
python dp/2_ml-to-npy.py data/all.mlu
