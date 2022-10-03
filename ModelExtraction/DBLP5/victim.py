import torch
import numpy as np
import networkx as nx
from dataset_loader import DBLPELoader,DBLPLoader
from TGCN.signal.train_test_split import temporal_signal_split, random_temporal_signal_split
import torch.optim.lr_scheduler as lr_scheduler
from models import ModelfreeGCN, VRecurrentGCN, GCN,LitDiffConvModel, DCRNN_RecurrentGCN, EvolveGCNO_RecurrentGCN, TGCN_RecurrentGCN, A3TGCN_RecurrentGCN
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from torch.utils.data import DataLoader
#from sampler import SubsetSequentialSampler

def DBLP5_victim_model(args, victim_type):
    url = str(victim_type)+'_victim_model'
    if victim_type =='DCRNN':
        model = DCRNN_RecurrentGCN(node_features=100,
                              num_classes=5)
    elif victim_type =='EVOLVEGCNO':
        model = EvolveGCNO_RecurrentGCN(node_features=100,
                                   num_classes=5)
    elif victim_type =='TGCN':
        model = TGCN_RecurrentGCN(node_features=100,
                                   num_classes=5)
    elif victim_type =='A3TGCN':
        model = A3TGCN_RecurrentGCN(node_features=100,
                                   num_classes=5)

    if torch.cuda.is_available():
        model = model.cuda()
    #exit()
    #model = RecurrentGCN(node_features=4)
    if os.path.exists(url) == False:
        file = open(url, 'w')
    if os.path.getsize(url) > 0:
        print('Saved victim model is loaded')
        weights = torch.load(f=url)
        model.load_state_dict(weights['victim_model'], strict=False)
        #print(weights['victim_model'])
        return model
    print('The data of victim_model is already for loading')
    loader = DBLPLoader('DBLP5')
    dataset = loader.get_dataset()
    print('The data of victim_model is loaded')
    use_cuda = torch.cuda.is_available()
    data_unlabeled = dataset
    # train_loader = dataset
    #print(torch.tensor(dataset.features).shape)  # [10, 6606, 100]
    train_loader, test_loader = temporal_signal_split(dataset, 0.8)
    #print(train_loader.features)
    #print(train_loader.targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60])
    model.train()
    for epoch in tqdm(range(4000)):
        cost = 0
        for time, snapshot in enumerate(dataset):
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().cuda()
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.cuda()
            edge_weight = snapshot.edge_attr.cuda()
            y_hat = model(x, edge_index, edge_weight)
            a= criterion(y_hat, labels)
            cost = cost + criterion(y_hat, labels)
            #print(cost)
        cost = cost / (time + 1)
        print('The '+str(epoch)+' training loss is '+str(cost))
        #print(cost)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    model.eval()
    cost = 0
    for time, snapshot in enumerate(test_loader):
        y = snapshot.y
        y = y.numpy()
        y = np.argmax(y, axis=1)
        labels = torch.from_numpy(y).long().cuda()
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.cuda()
        edge_weight = snapshot.edge_attr.cuda()
        y_hat = model(x, edge_index, edge_weight)
        cost = cost + criterion(y_hat, labels)
    cost = cost / (time + 1)
    cost = cost.item()
    print("Cross_Entropy: {:.4f}".format(cost))
    torch.save({'victim_model': model.state_dict()}, f=url)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    objective_function(model, train_loader, device)
    objective_function(model, test_loader, device)


    return model


def objective_function(model, dataset, device):
    accuracy = 0
    total_time = 0
    for time, snapshot in enumerate(dataset):
        with torch.cuda.device(device=device):
            x = snapshot.x.to(device)
            y = snapshot.y
            y_labels = torch.argmax(y, axis=1).to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            victim_labels = torch.argmax(model.to(device)(x, edge_index, edge_attr).detach(), dim=1).long().clone().to(device)
            accuracy += torch.eq(y_labels, victim_labels).sum() / y_labels.shape[0]
            total_time += 1
    print('The accuracy result of rgcn is ' + str(accuracy / total_time))


# 1.0704 lr=0.01 DCRNN
#The accuracy result of rgcn is tensor(0.8425, device='cuda:0')
#The accuracy result of rgcn is tensor(0.8355, device='cuda:0')

# 1.1102 lr=0.008 EVOLVE
#The accuracy result of rgcn is tensor(0.7986, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7940, device='cuda:0')

#1.1184  lr=0.01 TGCN
#The accuracy result of rgcn is tensor(0.7910, device='cuda:0')
#The accuracy result of rgcn is tensor(0.7867, device='cuda:0')

#1.1215  lr=0.01 A3TGCN
# The accuracy result of rgcn is tensor(0.7910, device='cuda:0')
# The accuracy result of rgcn is tensor(0.7838, device='cuda:0')