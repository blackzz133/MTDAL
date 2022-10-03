import argparse
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN, EvolveGCNO, TGCN, A3TGCN
import pytorch_lightning as pl
#from torch.nn import Dropout
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import torch.nn as nn
#from ModelExtraction.StaticGraphTemporalSignal.config import *
from dataset_loader import DBLPELoader
from TGCN.signal import temporal_signal_split
#from ModelExtraction.active_learning.utils import *
#from ModelExtraction.active_learning import *
import random
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from config import *
class GCN(torch.nn.Module):
    def __init__(self, node_features):
        super(GCN, self).__init__()
        self.conv_layer = GCNConv(node_features, 32)
        self.conv_layer2 = GCNConv(32, 1)
        #self.linear = torch.nn.Linear(8, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.conv_layer(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv_layer2(h, edge_index, edge_weight)
        #h = self.linear(h)
        return h



class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h



class ModelfreeGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(ModelfreeGCN, self).__init__()
        self.conv_layer = GCNConv(in_channels=node_features,
                                  out_channels=node_features,
                                  improved=False,
                                  cached=False,
                                  normalize=True,
                                  add_self_loops=True,
                                  bias=False)
    def forward(self, x, edge_index, edge_weight):
        Weight = self.conv_layer.lin.weight
        self.conv_layer.lin.weight = Parameter(torch.eye(Weight.shape[0], Weight.shape[1]))
        h = self.conv_layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return h


class DCRNN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(DCRNN_RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)
        #self.recurrent2 = DCRNN(32, num_classes, 1)


    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        #h = F.relu(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return h

class EvolveGCNO_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(EvolveGCNO_RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNO(node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

class TGCN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(TGCN_RecurrentGCN, self).__init__()
        self.recurrent = TGCN(node_features, 32)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class A3TGCN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(A3TGCN_RecurrentGCN, self).__init__()
        self.recurrent = A3TGCN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x.view(x.shape[0],x.shape[1],1), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h





