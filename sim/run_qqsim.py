from subprocess import Popen, PIPE

import sys
import os

benchmark = sys.argv[1]
num = sys.argv[2]
dataset = "data_spec_q"

tr_file_name = dataset + "/" + benchmark + ".qq100m.tr"
aux_file_name = dataset + "/" + benchmark + ".qq100m.tra"
if not(os.path.exists(tr_file_name)):
  print("Cannot open trace", tr_file_name)
  sys.exit()
if not(os.path.exists(aux_file_name)):
  print("Cannot open aux trace.")
  sys.exit()

lat_model = "models/" + sys.argv[3]
if len(sys.argv) == 4:
  binary = "./sim/build/simulator_qq_com"
  cmd = [binary,
         tr_file_name,
         aux_file_name,
         lat_model,
         num]
elif len(sys.argv) == 5:
  binary = "./sim/build/simulator_qq_cla"
  cla_model = "models/" + sys.argv[4]
  cmd = [binary,
         tr_file_name,
         aux_file_name,
         lat_model,
         cla_model,
         num]
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
