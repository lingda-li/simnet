import sys
import os
import subprocess
from subprocess import Popen, PIPE
from plot_cpi_curves import plot_cpi_curves, plot_legend


if len(sys.argv) < 2:
  print("Wrong number of arguments")
  sys.exit()

for_slides = True
dataset = "data_spec_q"
#trace_prefix = ".qq100m"
#trace_prefix = ".ac100m"
trace_prefix = ".sp100m"

benchmark_list = [
  '503.bwaves',
  '507.cactuBSSN',
  '508.namd',
  '510.parest',
  '511.povray',
  '519.lbm',
  '521.wrf',
  '526.blender',
  '527.cam4',
  '538.imagick',
  '544.nab',
  '549.fotonik3d',
  '554.roms',
  '997.specrand\_f',
  '500.perlbench',
  '502.gcc',
  '505.mcf',
  '520.omnetpp',
  '523.xalancbmk',
  '525.x264',
  '531.deepsjeng',
  '541.leela',
  '548.exchange2',
  '557.xz',
  '999.specrand\_i'
]

first = True

for benchmark in benchmark_list:
  args = [benchmark]
  for model in sys.argv[1:]:
    if 'specrand' in benchmark:
      benchmark_name = benchmark.replace('\\', '') + 'r'
    else:
      benchmark_name = benchmark + '_r'
    ipc_name = dataset + '/' + benchmark_name + trace_prefix + '.tr_' + model + '.ipc'
    args.append(ipc_name)
  plot_cpi_curves(args, for_slides)
  if not for_slides:
    output_name = 'fig/' + benchmark_name[0:3] + '_cpis.pdf'
    pdf_cmd = ['pdfcrop', output_name, output_name]
    process = subprocess.call(pdf_cmd)
  if first:
    first = False
    plot_legend(args, for_slides)
    if not for_slides:
      pdf_cmd = ['pdfcrop', 'fig/cpis_legend.pdf', 'fig/cpis_legend.pdf']
      process = subprocess.call(pdf_cmd)
