import numpy as np
import matplotlib.pyplot as plt
from os.path import basename
from os.path import splitext
import sys


def extract_data(filename, interval=10, skip=5000):
  data = np.genfromtxt(filename, delimiter=',', names=['lat'])
  size = 10000 - skip
  assert size % interval == 0 and interval > 0
  x = range(0,int(size/interval))
  y = data['lat'][skip::interval]
  for i in range(1, interval):
    y += data['lat'][skip+i::interval]
  y /= 1000 * interval
  return x, y


if (len(sys.argv) < 3):
  print("Usage: ./plot.py <name> <file>")
  sys.exit(0)

output_name = sys.argv[1][0:3] + '_cpi'
is_true = False
fig_h = 6
color = 'y'
if 'true' in sys.argv[2]:
  is_true = True
  fig_h = 6.7
  output_name += '_true'
  color = 'b'
elif 'E1DNet' in sys.argv[2]:
  output_name += '_enet'
  color = 'g'
elif 'CNN3' in sys.argv[2]:
  output_name += '_3c'
  color = 'y'

font = {'size' : 35}
plt.rc('font', **font)
fig, ax = plt.subplots(figsize=(10, fig_h), dpi=100)

x, y = extract_data(sys.argv[2], 10)
ax.plot(x, y, color)
#ax.scatter(x, y, c='b')

#ax.set_xlabel('10^4 instructions')
if is_true:
  ax.set_title(sys.argv[1], fontdict={'size': 60})
ax.set_ylabel('CPI')
ax.set_xlim(0, 500)
#if sys.argv[1][0:3] == '505':
if np.amax(y) >= 5.95 or sys.argv[1][0:3] == '531' or sys.argv[1][0:3] == '505':
  ax.set_ylim(0, 8)
elif np.amax(y) > 3.6:
  ax.set_ylim(0, 6)
else:
  ax.set_ylim(0, 4)
ax.grid(True)

fig.tight_layout()
fig.savefig('../fig/' + output_name + '.pdf')
plt.show()
