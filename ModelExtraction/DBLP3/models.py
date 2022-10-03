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
    def __init__(self, node_features, num_classes):
        super(GCN, self).__init__()
        self.node_features = node_features
        self.num_classes = num_classes
        self.conv_layer = GCNConv(node_features, 32)
        #self.linear = torch.nn.Linear(32, num_classes)
        self.conv_layer2 = GCNConv(32, num_classes)

    def forward(self, x, edge_index, edge_weight):
        h = self.conv_layer(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv_layer2(h, edge_index, edge_weight)
        #h = self.linear(h)
        return F.softmax(h, dim=1)

class DCRNN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(DCRNN_RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)
        #self.recurrent2 = DCRNN(32, num_classes, 1)


    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1)

class EvolveGCNO_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(EvolveGCNO_RecurrentGCN, self).__init__()
        self.recurrent = EvolveGCNO(node_features)
        self.linear = torch.nn.Linear(node_features, num_classes)
        #self.recurrent2 = EvolveGCNO(num_classes)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
       #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1)

class TGCN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(TGCN_RecurrentGCN, self).__init__()
        self.recurrent = TGCN(node_features, 32)
        #self.recurrent2 = TGCN(32, num_classes)
        self.linear = torch.nn.Linear(32, num_classes)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1)


class A3TGCN_RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(A3TGCN_RecurrentGCN, self).__init__()
        self.recurrent = A3TGCN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x.view(x.shape[0],x.shape[1],1), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return F.softmax(h, dim=1)







class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)
        #self.recurrent2 = DCRNN(32, num_classes, 1)


    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        #h = F.relu(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1)




class VRecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, num_classes):
        super(VRecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, num_classes)
        #self.recurrent2 = DCRNN(32, num_classes, 1)


    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        #h = F.relu(h)
        #h = self.recurrent2(h, edge_index, edge_weight)
        return F.softmax(h, dim=1)


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
        #Weight = self.conv_layer.weight
        #self.conv_layer.weight = Parameter(torch.eye(Weight.shape[0], Weight.shape[1]))
        #h = self.conv_layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return h


class LitDiffConvModel(pl.LightningModule):
    def __init__(self, node_features, num_classes):
        super().__init__()

        self.recurrent = DCRNN(node_features, 8, 1)
        self.recurrent2 = DCRNN(8, num_classes, 1)
        self.dropuout = torch.nn.Dropout(p=0.5)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-3)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        x = train_batch.x
        #y = train_batch.y.view(-1, 1)
        y = train_batch.y
        y = y.cpu().numpy()
        #print(type(y))
        #print(y[1])
        y = np.argmax(y, axis=1)
        y = torch.from_numpy(y).long().cuda()
        #print(y)
        edge_index = train_batch.edge_index
        edge_weight = train_batch.edge_attr
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.dropuout(h)
        h = self.recurrent2(h, edge_index, edge_weight)
        h = F.softmax(h, dim=1)
        loss = F.cross_entropy(h, y)
        #print('train_loss is :'+str(loss))
            #F.mse_loss(h, y)
        return loss
    def validation_step(self, val_batch, batch_idx):
        x = val_batch.x
        #y = val_batch.y.view(-1, 1)
        y = val_batch.y
        y = y.cpu().numpy()
        y = np.argmax(y, axis=1)
        y = torch.from_numpy(y).long().cuda()
        #print(y[1])
        #print(type(y))
        edge_index = val_batch.edge_index
        edge_weight = val_batch.edge_attr
        h = self.recurrent(x, edge_index, edge_weight)
        h = self.dropuout(h)
        h = F.relu(h)
        h = self.recurrent2(h, edge_index, edge_weight)
        h = F.softmax(h, dim=1)
        #exit()
        #print(h)
        #print(y)
        loss = F.cross_entropy(h,y)
        #print(2)
        #print(loss)
            #F.mse_loss(h, y)
        metrics = {'val_loss': loss}
        if time == []:
            time.append(0)
            loss_result.append(loss)
        else:
            time.append(len(time))
            loss_result.append(loss)
        time_cpu = torch.tensor(time, device='cpu').numpy().tolist()
        loss_result_cpu = torch.tensor(loss_result, device='cpu').numpy().tolist()
        #batch_size = 1
        self.log_dict(metrics)
        return metrics


