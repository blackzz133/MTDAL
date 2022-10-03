import torch
import numpy as np
import networkx as nx
from dataset_loader import ChickenpoxDatasetLoader
from TGCN.signal.train_test_split import temporal_signal_split, random_temporal_signal_split
import torch.optim.lr_scheduler as lr_scheduler
from models import ModelfreeGCN, RecurrentGCN, GCN
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from torch.utils.data import DataLoader
#from sampler import SubsetSequentialSampler
from models import ModelfreeGCN,  GCN, DCRNN_RecurrentGCN, EvolveGCNO_RecurrentGCN, TGCN_RecurrentGCN, A3TGCN_RecurrentGCN
def Chickenpox_victim_model(args,victim_type):
    url = str(victim_type) + '_victim_model'
    if victim_type == 'DCRNN':
        model = DCRNN_RecurrentGCN(node_features=4)
    elif victim_type == 'EVOLVEGCNO':
        model = EvolveGCNO_RecurrentGCN(node_features=4)
    elif victim_type == 'TGCN':
        model = TGCN_RecurrentGCN(node_features=4)
    elif victim_type == 'A3TGCN':
        model = A3TGCN_RecurrentGCN(node_features=4)

    if torch.cuda.is_available():
        model = model.cuda()
    # exit()
    # model = RecurrentGCN(node_features=4)
    if os.path.exists(url) == False:
        file = open(url, 'w')
    if os.path.getsize(url) > 0:
        print('Saved victim model is loaded')
        weights = torch.load(f=url)
        model.load_state_dict(weights['victim_model'], strict=False)
        # print(weights['victim_model'])
        return model

    print('The data of victim_model is already for loading')
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    print(torch.tensor(dataset.features).shape)
    print(torch.tensor(dataset.targets).shape)
    print(dataset.targets[0][0])
    print('The data of victim_model is loaded')
    use_cuda = torch.cuda.is_available()
    data_unlabeled = dataset
    # train_loader = dataset
    train_loader, test_loader = temporal_signal_split(dataset, 0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss(reduction='mean').cuda()
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60])
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    model.train()
    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(dataset):
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.cuda()
            edge_attr = snapshot.edge_attr.cuda()
            y = snapshot.y.cuda()
            y_hat = model(x, edge_index, edge_attr).view(-1).cuda()
            cost = cost + criterion(y, y_hat)
            #cost = cost+torch.mean((y_hat - y) ** 2)
        cost = cost / (time + 1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        #scheduler.step()
        print('The '+str(epoch)+' training loss is '+str(cost))
        #print(cost)
    #model = model.to('cpu')
    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_loader):
        y_hat = model(snapshot.x.cuda(), snapshot.edge_index.cuda(), snapshot.edge_attr.cuda()).view(-1)
        cost = cost + criterion(snapshot.y.cuda(), y_hat)
    cost = cost / (time + 1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))
    torch.save({'victim_model': model.state_dict()}, f=url)
    return model


