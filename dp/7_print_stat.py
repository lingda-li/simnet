import sys
import numpy as np

fs = np.load(sys.argv[1])

for i in range(39):
  print(np.sqrt(fs['all_var'][i]))

print()

for i in range(39):
  print(fs['all_mean'][i])
