from helper import *
from torch_geometric.nn.conv import HypergraphConv
import torch.nn as nn

class HCHA(nn.Module):
    """
    This model is proposed by "Hypergraph Convolution and Hypergraph Attention" (in short HCHA) and its convolutional layer
    is implemented in pyg.
    """

    def __init__(self, params):
        super(HCHA, self).__init__()

        self.p = params
        self.num_layers = self.p.hyper_layers
        self.dropout = self.p.dropout  # Note that default is 0.6
        self.symdegnorm = self.p.HyperAtt

#         Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphConv(self.p.hyper_init_dim,
                                         self.p.hyper_hidden, self.p.HyperAtt))
        for _ in range(self.num_layers-2):
            self.convs.append(HypergraphConv(
                self.p.hyper_hidden, self.p.hyper_hidden, self.symdegnorm))

        # Output heads is set to 1 as default
        self.convs.append(HypergraphConv(
            self.p.hyper_hidden, self.p.embed_dim, self.symdegnorm))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward_base(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.convs[-1](x, edge_index))

        return x