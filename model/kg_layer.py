from helper import *
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn

#compGCN卷积实现
class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x, params=None):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.act = act
        self.device = None

        self.w_loop = get_param((in_channels, out_channels))
        self.w_in = get_param((in_channels, out_channels))
        self.w_out = get_param((in_channels, out_channels))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels))

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, rel_embed):
        if self.device is None:
            self.device = edge_index.device

        # rel_embed
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        # 三种边分开传递一次然后聚合
        in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,
                                edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate(self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,
                                  edge_norm=None, mode='loop')
        out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                 edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.p.bias: out = out + self.bias
        out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]  # Ignoring the self loop inserted

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

#compGCNBasis卷积实现
class CompGCNConvBasis(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, num_bases, act=lambda x: x, cache=True, params=None):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.act = act
        self.device = None
        self.cache = cache  # Should be False for graph classification tasks

        self.w_loop = get_param((in_channels, out_channels));
        self.w_in = get_param((in_channels, out_channels));
        self.w_out = get_param((in_channels, out_channels));

        self.rel_basis = get_param((self.num_bases, in_channels))
        self.rel_wt = get_param((self.num_rels * 2, self.num_bases))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels));

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        self.in_norm, self.out_norm,
        self.in_index, self.out_index,
        self.in_type, self.out_type,
        self.loop_index, self.loop_type = None, None, None, None, None, None, None, None

        if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, edge_norm=None, rel_embed=None):
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.mm(self.rel_wt, self.rel_basis)  # 基向量矩阵乘权重矩阵得到所有关系的表征
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        if not self.cache or self.in_norm == None:
            self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
            self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

            # 在进行图神经网络搭建时构造的自连边
            self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
            self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)

            self.in_norm = self.compute_norm(self.in_index, num_ent)  # 对入节点进行正则化，需要再研究
            self.out_norm = self.compute_norm(self.out_index, num_ent)

        # w为了对三种类型的边使用不同的关系权重矩阵，这里把卷积分成了三个部分
        in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,
                                edge_norm=self.in_norm, mode='in')
        loop_res = self.propagate(self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,
                                  edge_norm=None, mode='loop')
        out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                 edge_norm=self.out_norm, mode='out')
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.p.bias: out = out + self.bias
        if self.b_norm: out = self.bn(out)

        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]

    def rel_transform(self, ent_embed, rel_embed):
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    # 研究一下edge_norm是什么
    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        print('compute_norm中的row形状', row.shape())
        print('compute_norm中的col形状', col.shape())
        edge_weight = torch.ones_like(row).float()  # 生成与input形状相同、元素全为1的张量
        deg = scatter_add(edge_weight, row, dim=0,
                          dim_size=num_ent)  # Summing number of weights of the edges [Computing out-degree] [Should be equal to in-degree (undireted graph)]
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

#compGCN整体实现
class CompGCN(nn.Module):
    def __init__(self, edge_index, edge_type, num_rel, params=None):#传入所需的全部kg数据
        super(CompGCN, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim  # gcn输出向量的维度
        self.device = self.edge_index.device

        if self.p.num_bases > 0:
            self.init_rel = get_param((self.p.num_bases, self.p.init_dim))
        else:
            self.init_rel = get_param((num_rel * 2, self.p.init_dim))

        if self.p.num_bases > 0:
            # in_channels, out_channels, num_rels, num_bases, act=lambda x:x, cache=True, params=None
            self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act,
                                          params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act,
                                     params=self.p) if self.p.gcn_layer == 2 else None
        else:
            self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act,
                                     params=self.p) if self.p.gcn_layer == 2 else None

        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))  # 这里注册了SELF.BIAS

    #取出所需org和author表征
    def forward_base(self, kg_node_emb, org, drop1, drop2):

        r = self.init_rel
        x, r = self.conv1(kg_node_emb, self.edge_index, self.edge_type, rel_embed=r)
        x = drop1(x)
        x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) if self.p.gcn_layer == 2 else (x, r)
        x = drop2(x) if self.p.gcn_layer == 2 else x

        org_emb = torch.index_select(x, 0, org)
        print()

        return org_emb, x