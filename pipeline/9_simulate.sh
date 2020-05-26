#!/bin/sh

#module load anaconda2
#source activate topaz

#tr_file_name = sys.argv[1]
#aux_file_name = sys.argv[2]
#models_dir = sys.argv[4]
#var_txt_file = sys.argv[3]


../sim/build/simulator_ground_truth $1.tr $1.tra converted_models/1.pt.pt stats/var.txt | tee simoutput.txt

#cd ../sim
python ../sim/runsimulations.py $1.tr $1.tra converted_models/  stats/var.txt
