from subprocess import Popen, PIPE

import sys
import os

benchmark = sys.argv[1]
num = sys.argv[2]
dataset = sys.argv[3]
rob = sys.argv[4]
trace_prefix = ".qq100m"
dataset = "data_spec_robreg/" + dataset + "/" + benchmark + trace_prefix

tr_file_name = dataset + ".tr"
aux_file_name = dataset + ".tra"
if not(os.path.exists(tr_file_name)):
  print("Cannot open trace", tr_file_name)
  sys.exit()
if not(os.path.exists(aux_file_name)):
  print("Cannot open aux trace.")
  sys.exit()

lat_model = "models/" + sys.argv[5]
if len(sys.argv) == 6:
  #binary = "./sim/build/simulator_rr_com"
  binary = "./sim/build/simulator_rr_ground_truth"
  cmd = [binary,
         tr_file_name,
         aux_file_name,
         lat_model,
         num,
         rob]
else:
  print("Illegal arguments")
  sys.exit()

print("Executing  %s" % " ".join(cmd))
process = Popen(cmd, stdout=PIPE)
output, err = process.communicate()
exit_code = process.wait()
print(output.decode("utf-8"),end='')
with open("res/all.txt","a") as f:
  f.write(output.decode("utf-8"))
