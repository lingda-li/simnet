#!/bin/sh

#module load anaconda2
#source activate topaz

cd ../data_spec
python ../dp/4_combine.py normall_train_test_1m.ml*npz
