import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from os.path import basename
from os.path import splitext
import sys
import pylab


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


def plot_cpi_curves(args, for_slides=False):
  output_name = args[0][0:3] + '_cpis'
  fig_h = 6
  mpl.rcParams['text.usetex'] = True
  font = {'size' : 36}
  plt.rc('font', **font)
  fig, ax = plt.subplots(figsize=(10, fig_h), dpi=100)
  #colors = cm.rainbow(np.linspace(0, 1, 2*(len(args) - 1)-1))
  colors = cm.rainbow(np.linspace(0, 1, len(args) - 1))
  #colors = np.concatenate((colors[0::2], colors[1::2]), axis=0)
  color = iter(colors)
  for i in range(1, len(args)):
    file_name = args[i]
    c = next(color)

    x, y = extract_data(file_name, 1000, 100)
    if for_slides:
      if 'true' in file_name:
        ax.plot(x, y, c=c, linewidth=2.5)
      else:
        ax.plot(x, y, c=c, linewidth=2.5, alpha=0.0)
    else:
      ax.plot(x, y, c=c, linewidth=2.5)
    if 'true' in file_name:
      true_y = y
    elif not for_slides:
      #c = next(color)
      ax.plot(x, y - true_y, c=c, linewidth=2.5, linestyle='dotted')

  #ax.scatter(x, y, c='b')
  ax.set_xlabel('Million instructions')
  ax.set_xlim(0, 100)
  ax.set_title(args[0][4:], fontdict={'size': 48})
  ax.set_ylabel('CPI')
  #ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0, ncol=1)
  #ax.grid(True)
  ax.yaxis.grid(True)

  fig.tight_layout()
  if for_slides:
    fig.savefig('slides/' + output_name + '.png')
  else:
    fig.savefig('fig/' + output_name + '.pdf')
  #plt.show()
  plt.close()


def plot_legend(args, for_slides=False):
  mpl.rcParams['text.usetex'] = True
  font = {'size' : 36}
  plt.rc('font', **font)
  fig, ax = plt.subplots(figsize=(30, 6), dpi=100)
  #colors = cm.rainbow(np.linspace(0, 1, 2*(len(args) - 1)-1))
  colors = cm.rainbow(np.linspace(0, 1, len(args) - 1))
  #colors = np.concatenate((colors[0::2], colors[1::2]), axis=0)
  color = iter(colors)
  for i in range(1, len(args)):
    file_name = args[i]
    c = next(color)
    if 'true' in file_name:
      label = 'gem5'
    elif 'E1DNet' in file_name:
      label = 'RB7'
    elif 'CNN3' in file_name:
      #str_idx = file_name.find('CNN')
      #label = file_name[str_idx:str_idx+9]
      label = 'C3'
    elif 'CNN5' in file_name:
      label = '5C'
    elif 'CNN7' in file_name:
      label = 'C7'
    elif 'InsLSTM' in file_name:
      label = 'LSTM2'
    else:
      assert 0

    x, y = extract_data(file_name, 500, 100)
    ax.plot(x, y, c=c, linewidth=2.5, label=label)
    if 'true' in file_name:
      true_y = y
    elif not for_slides:
      #c = next(color)
      label += ' error'
      ax.plot(x, y - true_y, c=c, linewidth=2.5, linestyle='dotted', label=label)

  ax.legend(bbox_to_anchor=(0.5, 1.02), loc='upper center', borderaxespad=0, ncol=2*(len(args) - 1)-1)

  figlegend = plt.figure(figsize=(60, 2), dpi=100)
  axlegend = figlegend.add_subplot(111)
  #axlegend.legend(*ax.get_legend_handles_labels(), loc='center')
  legend = axlegend.legend(*ax.get_legend_handles_labels(), loc='center', borderaxespad=0, ncol=2*(len(args) - 1)-1)
  legend.get_frame().set_linewidth(2)
  legend.get_frame().set_edgecolor("black")
  plt.gca().set_axis_off()
  figlegend.tight_layout()
  if for_slides:
    figlegend.savefig('slides/cpis_legend.png')
  else:
    figlegend.savefig('fig/cpis_legend.pdf')
  plt.close()


if __name__ == "__main__":
  if (len(sys.argv) < 3):
    print("Usage: ./plot.py <name> <file>")
    sys.exit(0)
  plot_cpi_curves(sys.argv[1:])
