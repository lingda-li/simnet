#!/bin/sh

#module load anaconda2
#source activate topaz

cd ../data_spec
python ../pipeline/combine_uncompressed.py normall_train_test_1m.ml*npz
