from subprocess import Popen, PIPE

import sys
import os
from plot_cpi_curves import plot_cpi_curves

if len(sys.argv) < 4:
  print("Wrong number of arguments")
  sys.exit()

num = sys.argv[1]
dataset = "data_spec_q"
#trace_prefix = ".qq100m"
#trace_prefix = ".ac100m"
trace_prefix = ".sp100m"

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
  args = [benchmark]
  for model in sys.argv[1:]:
    ipc_name = dataset + '/' + benchmark + trace_prefix + '.tr_' + model + '.ipc'
    args.append(ipc_name)
  plot_cpi_curves(args)
