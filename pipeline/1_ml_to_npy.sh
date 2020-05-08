#!/bin/bash

#module load anaconda2
#source activate topaz

cd ../data_spec
rm *.npz
python ../pipeline/ml-to-npy-0.py test_1m.ml
