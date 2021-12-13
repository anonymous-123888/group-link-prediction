import torch.nn as nn
from helper import *

class KGNodeEmbedding(nn.Module):

    def __init__(self, node_set_length, params):
        super(KGNodeEmbedding, self).__init__()
        self.p = params
        self.node_set_length = node_set_length
        self.embeddings = nn.ModuleList()#每一种类别用不同的W，同样的embedding
        self.w_rel = []
        for i in range(len(node_set_length)):
            self.embeddings.append(nn.Embedding(node_set_length[i],
                                             self.p.init_dim))
            self.w_rel.append(get_param((self.p.init_dim, self.p.init_dim)))

    def forward(self):
        out_x = []
        for i, emb_layer in enumerate(self.embeddings):
            index_tensor = torch.LongTensor([idx for idx in range(self.node_set_length[i])]).cuda() #[0,1,2,3,4,...len(set)]
            set_x = torch.matmul(emb_layer(index_tensor),self.w_rel[i])
            out_x.append(set_x)


        return torch.cat(out_x, 0)