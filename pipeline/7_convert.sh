#!/bin/sh

#module load anaconda2
#source activate topaz

rm ../data_spec/converted_models/*.pt
cd ../data_spec
python ../pipeline/convert.py
