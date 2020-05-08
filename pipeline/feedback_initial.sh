#!/bin/sh

rm best.pt best.pt.pt /raid/data/tflynn/gccvec/feedback_inputs.txt /raid/data/tflynn/gccvec/feedback_targets.txt /raid/data/tflynn/gccvec/errs.txt /raid/data/tflynn/gccvec/ticks.txt

#module load anaconda2
#source activate topaz

python train-from-one-file.py --no-cuda --save --epochs 1000000
python convert_only_best.py
../tools/simulator /raid/data/tflynn/gccvec/gccvec_1m.tr best.pt.pt /raid/data/tflynn/gccvec/mean.txt /raid/data/tflynn/gccvec/var.txt | tee -a simoutput.txt
