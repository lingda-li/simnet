import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from os.path import basename
from os.path import splitext
import sys


def extract_data(filename, interval=10, jump=1, skip=0):
  data = np.genfromtxt(filename, delimiter=',', names=['lat'])
  size = data['lat'].size - skip
  assert size % interval == 0 and interval > 0
  x = range(0,int(size/interval*jump),jump)
  y = data['lat'][skip::interval]
  for i in range(1, interval):
    y += data['lat'][skip+i::interval]
  y /= 1000 * interval
  return x, y


if (len(sys.argv) < 3):
  print("Usage: ./plot.py <name> <file>")
  sys.exit(0)

output_name = sys.argv[1][0:3] + '_cpis'
fig_h = 6
#fig_h = 6.7
font = {'size' : 35}
plt.rc('font', **font)
fig, ax = plt.subplots(figsize=(15, fig_h), dpi=100)
color = iter(cm.rainbow(np.linspace(0, 1, 2*(len(sys.argv) - 2)-1)))
#color = iter(cm.rainbow(np.linspace(0, 1, 1)))
for i in range(2, len(sys.argv)):
  file_name = sys.argv[i]
  c = next(color)
  if 'true' in file_name:
    label = 'true'
  elif 'E1DNet' in file_name:
    label = 'enet'
  elif 'CNN' in file_name:
    str_idx = file_name.find('CNN')
    label = file_name[str_idx:str_idx+9]
  
  x, y = extract_data(file_name, 200, 2)
  ax.plot(x, y, c=c, label=label)
  if 'true' in file_name:
    true_y = y
  else:
    c = next(color)
    label += ' diff'
    ax.plot(x, y - true_y, c=c, label=label)

#ax.scatter(x, y, c='b')

#ax.set_xlabel('10^4 instructions')
ax.set_title(sys.argv[1], fontdict={'size': 60})
ax.set_ylabel('CPI')
ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0, ncol=1)
#ax.set_xlim(0, 500)
##if sys.argv[1][0:3] == '505':
#if np.amax(y) >= 5.95 or sys.argv[1][0:3] == '531' or sys.argv[1][0:3] == '505':
#  ax.set_ylim(0, 8)
#elif np.amax(y) > 3.6:
#  ax.set_ylim(0, 6)
#else:
#  ax.set_ylim(0, 4)
ax.grid(True)

fig.tight_layout()
fig.savefig('fig/' + output_name + '.pdf')
#plt.show()
