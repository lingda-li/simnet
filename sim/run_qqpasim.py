from subprocess import Popen, PIPE

import sys
import os

if len(sys.argv) < 4:
  print("Wrong number of arguments")
  sys.exit()

num = sys.argv[1]
dataset = "data_spec_q"
#trace_prefix = ".qq100m"
#trace_prefix = ".ac100m"
trace_prefix = ".sp100m"

if sys.argv[2] == "lat":
  binary = "./sim/build/simulator_qq_pa_cpi"
elif sys.argv[2] == "gt":
  binary = "./sim/build/simulator_qq_pa_ground_truth"
elif sys.argv[2] == "com":
  binary = "./sim/build/simulator_qq_pa_com_cpi"
else:
  print("Illegal modes")
  sys.exit()
lat_model = "models/" + sys.argv[3]

cmd = [binary,
       lat_model,
       num]

benchmark_list = [
  '503.bwaves_r',
  '507.cactuBSSN_r',
  '508.namd_r',
  '510.parest_r',
  '511.povray_r',
  '519.lbm_r',
  '521.wrf_r',
  '526.blender_r',
  '527.cam4_r',
  '538.imagick_r',
  '544.nab_r',
  '549.fotonik3d_r',
  '554.roms_r',
  '997.specrand_fr',
  '500.perlbench_r',
  '502.gcc_r',
  '505.mcf_r',
  '520.omnetpp_r',
  '523.xalancbmk_r',
  '525.x264_r',
  '531.deepsjeng_r',
  '541.leela_r',
  '548.exchange2_r',
  '557.xz_r',
  '999.specrand_ir'
]

for benchmark in benchmark_list:
  tr_file_name = dataset + "/" + benchmark + trace_prefix + ".tr"
  aux_file_name = dataset + "/" + benchmark + trace_prefix + ".tra"
  if not(os.path.exists(tr_file_name)):
    print("Cannot open trace", tr_file_name)
    sys.exit()
  if not(os.path.exists(aux_file_name)):
    print("Cannot open aux trace.")
    sys.exit()
  cmd.extend([tr_file_name, aux_file_name])

os.environ["OMP_NUM_THREADS"] = str(len(benchmark_list))
print("Executing  %s" % " ".join(cmd))
process = Popen(cmd, stdout=PIPE)
output, err = process.communicate()
exit_code = process.wait()
print(output.decode("utf-8"),end='')
with open("res/sp.txt","a") as f:
  f.write(output.decode("utf-8"))
