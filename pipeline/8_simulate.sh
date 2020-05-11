#!/bin/sh

#module load anaconda2
#source activate topaz

#../sim/build/simulator_ground_truth /raid/data/tflynn/gccvec/gccvec_1m.tr /raid/data/tflynn/gccvec/converted_models/100.pt.pt /raid/data/tflynn/gccvec/mean.txt /raid/data/tflynn/gccvec/var.txt | tee simoutput.txt


python runsimulations.py
