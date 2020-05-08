#!/bin/sh

#module load anaconda2
#source activate topaz

python train-from-one-file.py --no-cuda --save --epochs 1000000 --resume best.pt
python convert_only_best.py
../tools/simulator /raid/data/tflynn/gccvec/gccvec_1m.tr best.pt.pt /raid/data/tflynn/gccvec/mean.txt /raid/data/tflynn/gccvec/var.txt | tee -a simoutput.txt
