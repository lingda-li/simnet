import sys
import numpy as np

#inst_length = 39
inst_length = 51

fs = np.load(sys.argv[1])

for i in range(inst_length):
  print(1.0 / np.sqrt(fs['all_var'][i]))

print()

for i in range(inst_length):
  print(fs['all_mean'][i])
