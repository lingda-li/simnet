import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from os.path import basename
from os.path import splitext
import sys


def extract_data(filename, interval=10, xmax=1, skip=0):
  data = np.genfromtxt(filename, delimiter=',', names=['lat'])
  size = data['lat'].size - skip
  assert size % interval == 0 and interval > 0
  jump = xmax / (size / interval)
  x = np.arange(0,xmax,jump)
  y = data['lat'][skip::interval]
  for i in range(1, interval):
    y += data['lat'][skip+i::interval]
  y /= 1000 * interval
  return x, y


def plot_cpi_curves(args):
  output_name = args[0][0:3] + '_cpis'
  fig_h = 6
  #fig_h = 6.7
  font = {'size' : 35}
  plt.rc('font', **font)
  fig, ax = plt.subplots(figsize=(15, fig_h), dpi=100)
  color = iter(cm.rainbow(np.linspace(0, 1, 2*(len(args) - 1)-1)))
  #color = iter(cm.rainbow(np.linspace(0, 1, 1)))
  for i in range(1, len(args)):
    file_name = args[i]
    c = next(color)
    if 'true' in file_name:
      label = 'true'
    elif 'E1DNet' in file_name:
      label = 'enet'
    elif 'CNN' in file_name:
      str_idx = file_name.find('CNN')
      label = file_name[str_idx:str_idx+9]
    
    x, y = extract_data(file_name, 200, 100)
    ax.plot(x, y, c=c, label=label)
    if 'true' in file_name:
      true_y = y
    else:
      c = next(color)
      label += ' diff'
      ax.plot(x, y - true_y, c=c, label=label)
  
  #ax.scatter(x, y, c='b')
  ax.set_xlabel('million instructions')
  ax.set_xlim(0, 100)
  ax.set_title(args[0], fontdict={'size': 60})
  ax.set_ylabel('CPI')
  ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0, ncol=1)
  ax.grid(True)
  
  fig.tight_layout()
  fig.savefig('fig/' + output_name + '.pdf')
  #plt.show()
  plt.close()


if __name__ == "__main__":
  if (len(sys.argv) < 3):
    print("Usage: ./plot.py <name> <file>")
    sys.exit(0)
  plot_cpi_curves(sys.argv[1:])
