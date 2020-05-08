#!/bin/sh

#module load anaconda2
#source activate topaz

cd ../data_spec
python ../pipeline/scale.py --save ./train_test_1m.ml*npz
