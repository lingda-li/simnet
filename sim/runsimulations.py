from subprocess import Popen, PIPE

import glob                                                                                                          
import os                                                                                                                                                     
files = glob.glob("../data_spec/converted_models/*.pt")   
files.sort(key=os.path.getmtime)

for fname in files:
    print(fname,end='')
    cmd = ["../sim/build/simulator", 
           "../data_spec/test_1m.tr",
           fname,
           "../data_spec/var.txt"]
    print("Executing  %s" % " ".join(cmd))
    process = Popen(cmd, stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    print(output.decode("utf-8"),end='')
    with open("simoutput.txt","a") as f: f.write(output.decode("utf-8"))
