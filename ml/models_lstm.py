import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from cfg_lstm import seq_length, input_length, tgt_length


class SeqLSTM(nn.Module):
  def __init__(self, nhidden, nlayers, nembed=0, gru=False, bi=False, norm=False):
    super(SeqLSTM, self).__init__()

    if nembed != 0:
      self.embed = True
      self.inst_embed = nn.Linear(input_length, nembed)
      nin = nembed
    else:
      self.embed = False
      nin = input_length
    self.bi = bi
    self.norm = norm
    if norm:
      #self.inst_norm = nn.LayerNorm(seq_length)
      self.inst_norm = nn.LayerNorm([seq_length, nin])
    if gru:
      self.lstm = nn.GRU(nin, nhidden, nlayers, batch_first=True, bidirectional=bi)
    else:
      self.lstm = nn.LSTM(nin, nhidden, nlayers, batch_first=True, bidirectional=bi)
    if bi:
      nhidden *= 2
    self.linear = nn.Linear(nhidden, tgt_length)

  #def init_hidden(self):
  #  # type: () -> Tuple[nn.Parameter, nn.Parameter]
  #  return (
  #    nn.Parameter(torch.zeros(1, 1, nhidden, requires_grad=True)),
  #    nn.Parameter(torch.zeros(1, 1, nhidden, requires_grad=True)),
  #  )

  def forward(self, x):
    if self.embed:
      x = self.inst_embed(x)
    if self.norm:
      #x = self.inst_norm(x.transpose(1, 2)).transpose(1, 2)
      x = self.inst_norm(x)
    x, _ = self.lstm(x)
    x = self.linear(x[:, -1, :])
    return x
