#!/bin/bash

#module load anaconda2
#source activate topaz

cd ../data_spec
rm *.npz
python ../dp/2_ml-to-npy.py test_1m.ml
