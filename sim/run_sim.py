from subprocess import Popen, PIPE

import sys
import os

benchmark = sys.argv[1]
dataset = sys.argv[2]
num = sys.argv[3]

tr_file_name = dataset + "/" + benchmark + ".10m.tr"
aux_file_name = dataset + "/" + benchmark + ".10m.tra"
var_txt_file = dataset + "/var.txt"
if not(os.path.exists(tr_file_name)):
  print("Cannot open trace", tr_file_name)
  sys.exit()
if not(os.path.exists(aux_file_name)):
  print("Cannot open aux trace.")
  sys.exit()
if not(os.path.exists(var_txt_file)):
  print("Cannot open var.")
  sys.exit()

lat_model = "models/" + sys.argv[4]
if len(sys.argv) == 5:
  if "_com" in lat_model:
    binary = "./sim/build/simulator_q_com"
  else:
    binary = "./sim/build/simulator_q"
  cmd = [binary,
         tr_file_name,
         aux_file_name,
         lat_model,
         var_txt_file,
         num]
elif len(sys.argv) == 6:
  binary = "./sim/build/simulator_q_cla"
  cla_model = "models/" + sys.argv[5]
  cmd = [binary,
         tr_file_name,
         aux_file_name,
         lat_model,
         cla_model,
         var_txt_file,
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
