import numpy as np
truth=np.fromfile("libsim.bin",dtype=np.float32)
custom=np.fromfile("libcustom.bin",dtype=np.float32)
truth= truth.reshape(-1,111,51)
custom= custom.reshape(-1,111,51)
for i in range(custom.shape[0]):
    if (np.array_equal(truth[i],custom[i])==False):
        print(i)
        print(np.array_equal(truth[i],custom[i]))
import ipdb; ipdb.set_trace()
