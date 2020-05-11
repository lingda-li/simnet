#!/bin/sh

#module load anaconda2
#source activate topaz

cd ..

python dp/make_sim_input.py data_spec/test_1m.ml -1 > data_spec/test_1m.tr
