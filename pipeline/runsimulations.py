from subprocess import Popen, PIPE

import glob                                                                                                          
import os                                                                                                                                                     
files = glob.glob("/raid/data/tflynn/gccvec/converted_models/*.pt")   
files.sort(key=os.path.getmtime)

for fname in files:
    print(fname,end='')
    cmd = ["../tools/simulator", 
           "/raid/data/tflynn/gccvec/gccvec_1m.tr",
           fname,
           "/raid/data/tflynn/gccvec/mean.txt",
           "/raid/data/tflynn/gccvec/var.txt"]
    print("Executing  %s" % " ".join(cmd))
    process = Popen(cmd, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    print(output.decode("utf-8"),end='')
    with open("simoutput.txt","a") as f: f.write(output.decode("utf-8"))
