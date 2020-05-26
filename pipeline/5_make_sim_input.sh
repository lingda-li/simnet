#!/bin/sh

#module load anaconda2
#source activate topaz

python ../dp/make_sim_input.py $1.ml -1 stats/statsall.npz > $1.tr

../dp/buildSim $1.txt 1000 > $1.tra
