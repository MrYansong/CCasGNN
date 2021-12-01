#encoding: utf-8

import torch
from torch_geometric.nn import GCNConv, GATConv
from math import sqrt



class Positional_GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, location_embedding_dim, filters_1, filters_2, dropout):
        super(Positional_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.dropout = dropout
        self.location_embedding_dim = location_embedding_dim
        self.setup_layers()
    def setup_layers(self):
        self.GAT_1 = GATConv(in_channels=self.in_channels,out_channels=self.filters_1, heads=self.n_heads, dropout=0.1)
        self.GAT_2 = GATConv(in_channels=self.filters_1 * self.n_heads + self.location_embedding_dim, out_channels=self.out_channels, heads=self.n_heads, dropout=0.1, concat=False)
    def forward(self, edge_indices, features, location_embedding):
        features = torch.cat((features, location_embedding), dim=-1)
        features = self.GAT_1(features, edge_indices)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = torch.cat((features, location_embedding), dim=-1)
        features = self.GAT_2(features, edge_indices)
        return features



class Positional_GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, location_embedding_dim, filters_1, filters_2, dropout):
        """
            GCN function
            :param args:  Arguments object.
            :param in_channel: Nodes' input feature dimensions
            :param out_channel: Nodes embedding dimension
            :param bais:
        """
        super(Positional_GCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.dropout = dropout
        self.location_embedding_dim = location_embedding_dim
        self.setup_layers()

    def setup_layers (self):
        self.convolution_1 = GCNConv(self.in_channels, self.filters_1)
        self.convolution_2 = GCNConv(self.filters_1 + self.location_embedding_dim, self.out_channels)

    def forward (self, edge_indices, features, location_embedding):
        """
        making convolution
        :param edge_indices: 2 * edge_number
        :param features: N * feature_size
        :return:
        """
        features = torch.cat((features, location_embedding), dim=-1)
        features = self.convolution_1(features, edge_indices)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.dropout,
                                               training=self.training)
        features = torch.cat((features, location_embedding), dim=-1)
        features = self.convolution_2(features, edge_indices)
        return features

class MultiHeadGraphAttention(torch.nn.Module):
    def __init__(self, num_heads, dim_in, dim_k, dim_v):
        super(MultiHeadGraphAttention, self).__init__()
        #"dim_k and dim_v must be multiple of num_heads"
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0
        self.num_heads = num_heads
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = torch.nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = torch.nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = torch.nn.Linear(dim_in, dim_v, bias=False)
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)
        self._nor_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh   #dim_k of each head
        dv = self.dim_v // nh
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1,2) # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1,2)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1,2)

        dist = torch.matmul(q, k.transpose(2,3)) * self._nor_fact # batch, nh, n, n
        # label = torch.where(dist == 0, torch.tensor(1), torch.tensor(0))
        # dist.data.masked_fill_(label, -float("inf"))
        dist = self.leaky_relu(dist) # batch, nh, n, n
        # dist = torch.where(torch.isnan(dist), torch.full_like(dist,0), dist)

        att = torch.matmul(dist, v) # batch, nh, n, dv
        att = att.transpose(1,2).reshape(batch, n, self.dim_v)
        return att


class dens_Net(torch.nn.Module):
    def __init__(self,dens_hiddensize, dens_dropout,  dens_inputsize, dens_outputsize):
        super(dens_Net, self).__init__()
        self.inputsize = dens_inputsize
        self.dens_hiddensize = dens_hiddensize
        self.dens_dropout = dens_dropout
        self.outputsize = dens_outputsize
        self.setup_layers()

    def setup_layers(self):
        self.dens_net = torch.nn.Sequential(
            torch.nn.Linear(self.inputsize, self.dens_hiddensize),
            torch.nn.Dropout(p=self.dens_dropout),
            torch.nn.Linear(self.dens_hiddensize, self.dens_hiddensize),
            torch.nn.Dropout(p=self.dens_dropout),
            torch.nn.Linear(self.dens_hiddensize, self.outputsize)
        )

    def forward(self, x1, x2):
        return torch.nn.functional.relu(self.dens_net(x1)), torch.nn.functional.relu(self.dens_net(x2))
        # return torch.nn.functional.relu(self.dens_net(x1))

class fuse_gate(torch.nn.Module):
    def __init__(self, batch_size, in_dim):
        super(fuse_gate, self).__init__()
        self.indim = in_dim
        self.batch_size = batch_size
        self.setup_layers()
    def setup_layers(self):
        self.omega = torch.nn.Parameter(torch.tensor([[0.5],[0.5]]))

    def forward(self, x):
        omega = self.omega.transpose(1,0)
        prediction = torch.matmul(omega, x)
        return prediction, self.omega[0], self.omega[1]
