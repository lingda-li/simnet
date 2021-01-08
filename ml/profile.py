import time
import torch
from ptflops import get_model_complexity_info
from models import *

#with torch.cuda.device(0):
#  #simnet = CNN3_F(2, 2, 64, 2, 1, 2, 128, 2, 0, 2, 256, 2, 0, 400)
#  #simnet = CNN3_F_P(22, 64, 2, 64, 2, 1, 2, 128, 2, 0, 2, 256, 2, 0, 400)
#  #simnet = CNN5_F(22, 2, 64, 2, 1, 2, 128, 2, 0, 2, 256, 2, 0, 2, 512, 2, 0, 2, 1024, 2, 1, 400)
#  #simnet = CNN7_F(22, 2, 64, 2, 1, 2, 128, 2, 0, 2, 256, 2, 0, 2, 512, 2, 0, 2, 1024, 2, 1, 2, 2048, 2, 0, 2, 4096, 2, 0, 400)
#  simnet = Efficient1DNet.from_name('e1d-b0')
#  macs, params = get_model_complexity_info(simnet, (context_length, inst_length), as_strings=True,
#                                           print_per_layer_stat=True, verbose=True)
#  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
input = torch.zeros(4096, context_length * inst_length).to(device)
simnet = Fusion1d(64).to(device)
start = time.time()
output = simnet(input)
end = time.time()

print(output.size())
print("Time:", end - start)
