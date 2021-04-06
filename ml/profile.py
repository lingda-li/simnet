import sys
import time
import torch
from utils import profile_model
from models import *


start = time.time()
model = eval(sys.argv[1])
profile_model(model)
end = time.time()
print("Time:", end - start)
