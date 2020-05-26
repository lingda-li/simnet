from subprocess import Popen, PIPE

import glob
import sys
import os

tr_file_name = sys.argv[1]
aux_file_name = sys.argv[2]
models_dir = sys.argv[3]
var_txt_file = sys.argv[4]

files = glob.glob("%s/*.pt" % models_dir)

files.sort(key=os.path.getmtime)

for fname in files:
    print(fname,end='')
    cmd = ["../sim/build/simulator", 
           tr_file_name,
           aux_file_name,
           fname,
           var_txt_file]
    print("Executing  %s" % " ".join(cmd))
    process = Popen(cmd, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    print(output.decode("utf-8"),end='')
    with open("simoutput.txt","a") as f: f.write(output.decode("utf-8"))
