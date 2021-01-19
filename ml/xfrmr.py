import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self,
                 ntoken,  # size of vocabulary
                 ninp,    # what dimension we want to embed our vocabulary
                 nhead,   # number of 'heads'
                 nhid,    # dimension of the feedforward network in nn.TransformerEncoder
                 nlayers, #number of encoder layers
                 dropout=0.5):
        print("ninp",ninp,
              "nhead",nhead,
              "nhid",nhid,
              "nlayers",nlayers)
        
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            print(mask.shape)
            print(mask)
            self.src_mask = mask
        print("Input shape is",src.shape)
        src = self.encoder(src) * math.sqrt(self.ninp)
        print("Encoder output shape is",src.shape)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) #5000 x 51
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #5000,1
        seq_length = 111 #10000
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(seq_length)/d_model))#26
        pe[:, 0::2] = torch.sin(position * div_term)
        dimen = pe[:,1::2].shape[-1]
        print("Dimen",dimen)
        pe[:, 1::2] = torch.cos(position * div_term)[:,:dimen]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CustomTransformerModel(nn.Module):

    def __init__(self,
                 ninp, # what dimension our input is
                 nhead, # number of 'heads'
                 nhid,  # dimension of the feedforward network in nn.TransformerEncoder
                 nlayers, #number of encoder layers
                 dropout=0.5):
        super(CustomTransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=ninp,
                                                 nhead=nhead,
                                                 dim_feedforward=nhid,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 2)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float(0.0))
        return mask
    
    def init_weights(self):
        initrange = 0.1

        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.transpose(0,1) # for multigpu
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask

        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        output = output.transpose(0,1) # for multigpu
        return output
