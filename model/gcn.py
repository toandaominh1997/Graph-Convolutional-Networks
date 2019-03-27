import torch
import torch.nn as nn 
import torch.nn.functional as F 
import math
import numpy as np
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weightadj = Parameter(torch.FloatTensor(2708, 2708))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weightadj.data = torch.ones((2708, 2708))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        adj = torch.mm(adj, self.weightadj)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhidden, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhidden)
        self.gc2 = GraphConvolution(nhidden, nhidden)
        self.gc3 = GraphConvolution(nhidden*2, nhidden)
        self.fc = nn.Linear(nhidden, nclass)
        self.dropout = dropout
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        feature1 = self.gc2(x, adj)
        feature2 = self.gc2(x, adj)
        x = torch.cat((feature1, feature2), dim=1)
        x = self.gc3(x, adj)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
