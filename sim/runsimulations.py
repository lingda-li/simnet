from subprocess import Popen, PIPE

import glob
import sys
import os

tr_file_name = sys.argv[1]
aux_file_name = sys.argv[2]

files = glob.glob("../data_spec/converted_models/*.pt")   
files.sort(key=os.path.getmtime)

for fname in files:
    print(fname,end='')
    cmd = ["../sim/build/simulator", 
           tr_file_name,
           aux_file_name,
           fname,
           "../data_spec/var.txt"]
    print("Executing  %s" % " ".join(cmd))
    process = Popen(cmd, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    print(output.decode("utf-8"),end='')
    with open("simoutput.txt","a") as f: f.write(output.decode("utf-8"))
