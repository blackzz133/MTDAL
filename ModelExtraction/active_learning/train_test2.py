import torch
from tqdm import tqdm
import numpy as np
from TGCN.signal.train_test_split import temporal_signal_split, random_temporal_signal_split
from sklearn.metrics import f1_score, mean_squared_error


def train2(models, criterions, optimizers, schedulers, dataloaders, labeled_set, num_epochs, epoch_loss, all_loss,
          device):
    criterion = criterions['mse']
    if 'RGCN' in models:
        network_type = 'RGCN'
        for epoch in tqdm(range(num_epochs), ncols=100):
            print(str(epoch) + ' epoch RGCN training is started')
            rgcn_loss = rgcn_train_epoch(models, network_type, criterion, optimizers['optimizer_rgcn'], dataloaders,
                                         labeled_set, epoch_loss, device)
            # print('training rgcn loss is '+str(rgcn_loss))
            schedulers['scheduler_rgcn'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                print('Test is started')
                rgcn_acc = test2(models, network_type, criterion, dataloaders, device, mode='test')
                all_loss['RGCN'] = rgcn_acc
                print('testing rgcn loss is ' + str(rgcn_acc) + ' in ' + str(epoch) + ' cycle')

    # exit()
    if 'GCN' in models:
        network_type = 'GCN'
        for epoch in tqdm(range(num_epochs), ncols=100):
            print(str(epoch) + ' epoch GCN training is started')
            gcn_loss = rgcn_train_epoch(models, network_type, criterion, optimizers['optimizer_gcn'], dataloaders,
                                        labeled_set, epoch_loss, device)
            # print('training gcn loss is ' + str(gcn_loss))
            schedulers['scheduler_gcn'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                gcn_acc = test2(models, network_type, criterion, dataloaders, device, mode='test')
                all_loss['GCN'] = gcn_acc
                print('testing gcn loss is ' + str(gcn_acc) + ' in ' + str(epoch) + ' cycle')

    if 'RGCN2' in models:
        network_type = 'RGCN2'
        for epoch in tqdm(range(num_epochs), ncols=100):
            print(str(epoch) + ' epoch RGCN2 training is started')
            rgcn_loss2 = rgcn_train_epoch(models, network_type, criterion, optimizers['optimizer_rgcn2'], dataloaders,
                                          labeled_set, epoch_loss, device)
            # print('training rgcn2 loss is '+str(rgcn_loss2))
            schedulers['scheduler_rgcn2'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                print('Test is started')
                rgcn_acc2 = test2(models, network_type, criterion, dataloaders, device, mode='test')
                all_loss['RGCN2'] = rgcn_acc2
                print('testing rgcn2 loss is ' + str(rgcn_acc2) + ' in ' + str(epoch) + ' cycle')
    if 'RGCN3' in models:
        network_type = 'RGCN3'
        for epoch in tqdm(range(num_epochs), ncols=100):
            print(str(epoch) + ' epoch RGCN3 training is started')
            rgcn_loss3 = rgcn_train_epoch(models, network_type, criterion, optimizers['optimizer_rgcn3'], dataloaders,
                                          labeled_set, epoch_loss, device)
            schedulers['scheduler_rgcn3'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                rgcn_acc3 = test2(models, network_type, criterion, dataloaders, device, mode='test')
                all_loss['RGCN3'] = rgcn_acc3
                print('testing rgcn3 loss is ' + str(rgcn_acc3) + ' in ' + str(epoch) + ' cycle')

    if 'RGCN4' in models:
        network_type = 'RGCN4'
        for epoch in tqdm(range(num_epochs), ncols=100):
            print(str(epoch) + ' epoch RGCN4 training is started')
            rgcn_loss4 = rgcn_train_epoch(models, network_type, criterion, optimizers['optimizer_rgcn4'], dataloaders,
                                          labeled_set, epoch_loss, device)
            schedulers['scheduler_rgcn4'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                rgcn_acc4 = test2(models, network_type, criterion, dataloaders, device, mode='test')
                all_loss['RGCN4'] = rgcn_acc4
                print('testing rgcn4 loss is ' + str(rgcn_acc4) + ' in ' + str(epoch) + ' cycle')

    return all_loss


def rgcn_train_epoch(models, network_type, criterion, optimizer, dataloaders, labeled_set, epoch_loss, device):
    victim_model = models['VICTIM'].to(device)
    new_labeled_set = torch.from_numpy(labeled_set)
    # train_loader = dataloaders['train']
    # print(type(models[network_type]))
    models[network_type] = models[network_type].to(device)
    models[network_type].train()
    rgcn_loss = 0
    for time, snapshot in enumerate(dataloaders['sim_train']):
        with torch.cuda.device(device=device):
            # labeled_set需要起作用
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            y = victim_model(x, edge_index, edge_attr).detach()[new_labeled_set[time]].view(-1).to(device)
            y_hat = models[network_type](x, edge_index, edge_attr)[new_labeled_set[time]].view(-1).to(device)
            rgcn_loss += criterion(y_hat, y)
    optimizer.zero_grad(0)
    rgcn_loss = rgcn_loss / (time + 1)
    rgcn_loss.backward()
    optimizer.step()

    '''
    if method == 'lloss':
        models['module'].train()
    global iters


    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            inputs = data[0].cuda()
            labels = data[1].cuda()
        optimizers['optimizer'].zero_grad()
        scores, _, features = models['backbone'](inputs)
        loss = criterion(scores, labels)
        loss.backward()
        optimizers['optimizer'].step()
    '''
    return rgcn_loss


'''
def gcn_train_epoch(models, network_type, criterion, optimizer, dataloaders, labeled_set, epoch_loss, device):
    victim_model = models['VICTIM']
    new_labeled_set = torch.from_numpy(labeled_set)
    #train_loader = dataloaders['train']
    models[network_type].train()
    gcn_loss = 0
    for time, snapshot in enumerate(dataloaders['sim_train']):
        with torch.cuda.device(device=device):
            x = snapshot.x.cuda()
            edge_index = snapshot.edge_index.cuda()
            edge_attr = snapshot.edge_attr.cuda()
            labels = victim_model(x, edge_index, edge_attr).detach()[labeled_set[time]]
            scores = models[network_type](x, edge_index, edge_attr)[labeled_set[time]]
            optimizer.zero_grad(0)
            gcn_loss = criterion(scores, labels)
    gcn_loss = gcn_loss/(time+1)
    gcn_loss.backward()
    optimizer.step()
'''


def test2(models, network_type, criterion, dataloaders, device, mode='val'):
    ####需要修改
    assert mode == 'val' or mode == 'test'
    model = models[network_type]
    model.eval()
    total_loss = 0
    for time, snapshot in enumerate(dataloaders['raw_test']):
        with torch.cuda.device(device=device):
            x = snapshot.x.to(device)
            y = snapshot.y.detach().to(device)
            edge_index = snapshot.edge_index.to(device)
            # print(snapshot.edge_attr)
            edge_attr = snapshot.edge_attr.to(device)
            y_hat= model(x, edge_index, edge_attr).detach().to(device)
            # print(scores.shape)
            loss = criterion(y_hat, y)
            total_loss += loss
    return total_loss / (time + 1)


# node regression
def rgcn_train_epoch2(models, network_type, criterion, optimizer, dataloaders, labeled_set, epoch_loss, device):
    victim_model = models['VICTIM'].to(device)
    new_labeled_set = torch.from_numpy(labeled_set)
    # train_loader = dataloaders['train']
    # print(type(models[network_type]))
    models[network_type] = models[network_type].to(device)
    models[network_type].train()
    rgcn_loss = 0
    for time, snapshot in enumerate(dataloaders['sim_train']):
        with torch.cuda.device(device=device):
            # labeled_set需要起作用
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            y_victim = victim_model(x, edge_index, edge_attr).detach().long()[new_labeled_set[time]].to(device)
            y_hat = models[network_type](x, edge_index, edge_attr)[new_labeled_set[time]].to(device)
            rgcn_loss += criterion(y_hat, y_victim)
    optimizer.zero_grad(0)
    rgcn_loss = rgcn_loss / (time + 1)
    rgcn_loss.backward()
    optimizer.step()

    return rgcn_loss


# node regression
def test2(models, network_type, criterion, dataloaders, device, mode='val'):
    ####需要修改
    assert mode == 'val' or mode == 'test'
    model = models[network_type]
    model.eval()
    total_loss = 0
    for time, snapshot in enumerate(dataloaders['raw_test']):
        with torch.cuda.device(device=device):
            # print('1')
            # print(snapshot.x)
            # print(snapshot.x.shape)
            x = snapshot.x.to(device)
            y_victim = snapshot.y.to(device)
            # print(labels)
            # print(snapshot.edge_index)
            edge_index = snapshot.edge_index.to(device)
            # print(snapshot.edge_attr)
            edge_attr = snapshot.edge_attr.to(device)
            y_hat = model(x, edge_index, edge_attr).detach().to(device)
            # print(scores.shape)
            loss = criterion(y_hat, y_victim)
            total_loss += loss
    return total_loss / (time + 1)


def LossPredLoss(input, target, margin, reduction='mean'):
    return 0


def objective_function2(models, dataloaders, device):
    dataset = dataloaders['sim_train']
    result = 0
    result2 = 0
    result3 = 0
    result4 = 0
    gcn_result =0
    total_time = 0
    for time, snapshot in enumerate(dataset):
        with torch.cuda.device(device=device):
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            victim_labels = models['victim'](x, edge_index, edge_attr).clone().detach().to(device)
            rgcn_labels = models['RGCN'](x, edge_index, edge_attr).clone().detach().to(device)
            rgcn2_labels = models['RGCN2'](x, edge_index, edge_attr).clone().detach().to(device)
            rgcn3_labels = models['RGCN3'](x, edge_index, edge_attr).clone().detach().to(device)
            rgcn4_labels = models['RGCN4'](x, edge_index, edge_attr).clone().detach().to(device)
            gcn_labels = models['GCN'](x, edge_index, edge_attr).clone().detach().to(device)
            result += mean_squared_error(victim_labels, rgcn_labels, average='binary')
            result2 += mean_squared_error(victim_labels, rgcn2_labels, average='binary')
            result3 += mean_squared_error(victim_labels, rgcn3_labels, average='binary')
            result4 += mean_squared_error(victim_labels, rgcn4_labels, average='binary')
            gcn_result += mean_squared_error(victim_labels, gcn_labels, average='binary')
            total_time += 1

    return result / total_time, result2 / total_time, result3 / total_time, result4 / total_time, gcn_result / total_time