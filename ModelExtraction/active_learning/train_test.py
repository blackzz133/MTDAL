import torch
from tqdm import tqdm
import numpy as np
from TGCN.signal.train_test_split import temporal_signal_split, random_temporal_signal_split
from sklearn.metrics import f1_score, mean_squared_error
import torch.nn.functional as F
#node classification
def train(models, criterions, optimizers, schedulers, dataloaders, labeled_set, num_epochs, all_loss, device):
    criterion = criterions['cross_entropy']

    if 'RGCN' in models:
        network_type = 'RGCN'
        for epoch in tqdm(range(num_epochs), ncols=100):
            #print(str(epoch) + ' epoch RGCN training is started')
            rgcn_loss = rgcn_train_epoch(models, network_type, criterion, optimizers['optimizer_rgcn'], dataloaders, labeled_set, device)
            #print('training rgcn loss is '+str(rgcn_loss))
            #schedulers['scheduler_rgcn'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                #print('Test is started')
                rgcn_acc = test(models, network_type, criterion, dataloaders, device, mode='test')
                #all_loss['RGCN'] = rgcn_acc
                print('testing rgcn loss is ' + str(rgcn_acc) + ' in ' + str(epoch) + ' cycle')
    if 'GCN' in models:
        network_type = 'GCN'
        for epoch in tqdm(range(num_epochs), ncols=100):
            #print(str(epoch) + ' epoch GCN training is started')
            gcn_loss = rgcn_train_epoch(models, network_type, criterion, optimizers['optimizer_gcn'], dataloaders, labeled_set, device)
            #print('training gcn loss is ' + str(gcn_loss))
            #schedulers['scheduler_gcn'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                gcn_acc = test(models, network_type, criterion, dataloaders, device, mode='test')
                #all_loss['GCN'] = gcn_acc
                print('testing gcn loss is ' + str(gcn_acc) + ' in ' + str(epoch) + ' cycle')

    if 'RGCN2' in models:
        network_type = 'RGCN2'
        for epoch in tqdm(range(num_epochs), ncols=100):
            #print(str(epoch) + ' epoch RGCN2 training is started')
            rgcn_loss2 = rgcn_train_epoch(models, network_type, criterion, optimizers['optimizer_rgcn2'], dataloaders, labeled_set, device)
            #print('training rgcn2 loss is '+str(rgcn_loss2))
            #schedulers['scheduler_rgcn2'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                #print('Test is started')
                rgcn_acc2 = test(models, network_type, criterion, dataloaders, device, mode='test')
                #all_loss['RGCN2'] = rgcn_acc2
                print('testing rgcn2 loss is ' + str(rgcn_acc2) + ' in ' + str(epoch) + ' cycle')
    if 'RGCN3' in models:
        network_type = 'RGCN3'
        for epoch in tqdm(range(num_epochs), ncols=100):
            #print(str(epoch) + ' epoch RGCN3 training is started')
            rgcn_loss3 = rgcn_train_epoch(models, network_type, criterion, optimizers['optimizer_rgcn3'], dataloaders, labeled_set, device)
            #schedulers['scheduler_rgcn3'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                rgcn_acc3 = test(models, network_type, criterion, dataloaders, device, mode='test')
                #all_loss['RGCN3'] = rgcn_acc3
                print('testing rgcn3 loss is ' + str(rgcn_acc3) + ' in ' + str(epoch) + ' cycle')

    if 'RGCN4' in models:
        network_type = 'RGCN4'
        for epoch in tqdm(range(num_epochs), ncols=100):
            #print(str(epoch) + ' epoch RGCN4 training is started')
            rgcn_loss4 = rgcn_train_epoch(models, network_type, criterion, optimizers['optimizer_rgcn4'], dataloaders, labeled_set, device)
            #schedulers['scheduler_rgcn4'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                rgcn_acc4 = test(models, network_type, criterion, dataloaders, device, mode='test')
                #all_loss['RGCN4'] = rgcn_acc4
                print('testing rgcn4 loss is ' + str(rgcn_acc4) + ' in ' + str(epoch) + ' cycle')
    all_loss['RGCN'], all_loss['RGCN2'], all_loss['RGCN3'], all_loss['RGCN4'], all_loss['GCN'] = objective_function(models, dataloaders['raw_test'], device)
    return all_loss



def rgcn_train_epoch(models, network_type, criterion, optimizer , dataloaders, labeled_set, device):
    victim_model = models['VICTIM'].to(device)
    new_labeled_set = torch.from_numpy(labeled_set)
    #train_loader = dataloaders['train']
    #print(type(models[network_type]))
    models[network_type] = models[network_type].to(device)
    models[network_type].train()
    rgcn_loss = 0
    for time, snapshot in enumerate(dataloaders['sim_train']):
        with torch.cuda.device(device=device):
            # labeled_set需要起作用
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            #labels = torch.argmax(victim_model(x, edge_index, edge_attr).detach(), dim=1).long().to(device)
            #scores = models[network_type](x, edge_index, edge_attr).to(device)

            labels = torch.argmax(victim_model(x, edge_index, edge_attr).detach(),dim=1).long()[new_labeled_set[time]].to(device)
            scores = models[network_type](x, edge_index, edge_attr)[new_labeled_set[time]].to(device)
            rgcn_loss += criterion(scores, labels)
    optimizer.zero_grad(0)
    rgcn_loss = rgcn_loss/(time+1)
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

def test(models, network_type, criterion, dataloaders, device, mode='val'):
    ####需要修改
    assert mode == 'val' or mode == 'test'
    model = models[network_type]
    model.eval()
    total_loss = 0
    for time, snapshot in enumerate(dataloaders['raw_test']):
        with torch.cuda.device(device=device):
            #print('1')
            #print(snapshot.x)
            #print(snapshot.x.shape)
            x = snapshot.x.to(device)
            labels = torch.argmax(snapshot.y, axis=1).long().detach().to(device)
            #print(labels)
            #print(snapshot.edge_index)
            edge_index = snapshot.edge_index.to(device)
            #print(snapshot.edge_attr)
            edge_attr = snapshot.edge_attr.to(device)
            scores = model(x, edge_index, edge_attr).detach().to(device)
            #print(scores.shape)
            loss = criterion(scores, labels)
            total_loss += loss
    return total_loss/(time+1)

# node regression
def train2(models, criterions, optimizers, schedulers, dataloaders, labeled_set, num_epochs, all_loss,
          device):
    criterion = criterions['mse']
    if 'RGCN' in models:
        network_type = 'RGCN'
        for epoch in tqdm(range(num_epochs), ncols=100):
            #print(str(epoch) + ' epoch RGCN training is started')
            rgcn_loss = rgcn_train_epoch2(models, network_type, criterion, optimizers['optimizer_rgcn'], dataloaders,
                                         labeled_set, device)
            # print('training rgcn loss is '+str(rgcn_loss))
            #schedulers['scheduler_rgcn'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                #print('Test is started')
                rgcn_acc = test2(models, network_type, criterion, dataloaders, device, mode='test')
                all_loss['RGCN'] = rgcn_acc
                #print('testing rgcn loss is ' + str(rgcn_acc) + ' in ' + str(epoch) + ' cycle')
    # exit()
    if 'GCN' in models:
        network_type = 'GCN'
        for epoch in tqdm(range(num_epochs), ncols=100):
            #print(str(epoch) + ' epoch GCN training is started')
            gcn_loss = rgcn_train_epoch2(models, network_type, criterion, optimizers['optimizer_gcn'], dataloaders,
                                        labeled_set, device)
            # print('training gcn loss is ' + str(gcn_loss))
            #schedulers['scheduler_gcn'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                gcn_acc = test2(models, network_type, criterion, dataloaders, device, mode='test')
                all_loss['GCN'] = gcn_acc
                #print('testing gcn loss is ' + str(gcn_acc) + ' in ' + str(epoch) + ' cycle')

    if 'RGCN2' in models:
        network_type = 'RGCN2'
        for epoch in tqdm(range(num_epochs), ncols=100):
            #print(str(epoch) + ' epoch RGCN2 training is started')
            rgcn_loss2 = rgcn_train_epoch2(models, network_type, criterion, optimizers['optimizer_rgcn2'], dataloaders,
                                          labeled_set, device)
            # print('training rgcn2 loss is '+str(rgcn_loss2))
            #schedulers['scheduler_rgcn2'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                #print('Test is started')
                rgcn_acc2 = test2(models, network_type, criterion, dataloaders, device, mode='test')
                all_loss['RGCN2'] = rgcn_acc2
                #print('testing rgcn2 loss is ' + str(rgcn_acc2) + ' in ' + str(epoch) + ' cycle')
    if 'RGCN3' in models:
        network_type = 'RGCN3'
        for epoch in tqdm(range(num_epochs), ncols=100):
            #print(str(epoch) + ' epoch RGCN3 training is started')
            rgcn_loss3 = rgcn_train_epoch2(models, network_type, criterion, optimizers['optimizer_rgcn3'], dataloaders,
                                          labeled_set, device)
            #schedulers['scheduler_rgcn3'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                rgcn_acc3 = test2(models, network_type, criterion, dataloaders, device, mode='test')
                all_loss['RGCN3'] = rgcn_acc3
                #print('testing rgcn3 loss is ' + str(rgcn_acc3) + ' in ' + str(epoch) + ' cycle')

    if 'RGCN4' in models:
        network_type = 'RGCN4'
        for epoch in tqdm(range(num_epochs), ncols=100):
            #print(str(epoch) + ' epoch RGCN4 training is started')
            rgcn_loss4 = rgcn_train_epoch2(models, network_type, criterion, optimizers['optimizer_rgcn4'], dataloaders,
                                          labeled_set, device)
            #schedulers['scheduler_rgcn4'].step()
            if epoch % 5 == 4 or epoch == num_epochs - 1:
                rgcn_acc4 = test2(models, network_type, criterion, dataloaders, device, mode='test')
                all_loss['RGCN4'] = rgcn_acc4
                #print('testing rgcn4 loss is ' + str(rgcn_acc4) + ' in ' + str(epoch) + ' cycle')
    all_loss['RGCN'], all_loss['RGCN2'], all_loss['RGCN3'], all_loss['RGCN4'], all_loss['GCN'] = objective_function(
        models, dataloaders['raw_test'], device)
    return all_loss

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
            y_victim = snapshot.y.view(-1).to(device)
            # print(labels)
            # print(snapshot.edge_index)
            edge_index = snapshot.edge_index.to(device)
            # print(snapshot.edge_attr)
            edge_attr = snapshot.edge_attr.to(device)
            y_hat = model(x, edge_index, edge_attr).detach().view(-1).to(device)
            # print(scores.shape)
            loss = criterion(y_hat, y_victim)
            total_loss += loss
    return total_loss / (time + 1)

def rgcn_train_epoch2(models, network_type, criterion, optimizer, dataloaders, labeled_set, device):
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
            #y = victim_model(x, edge_index, edge_attr).detach()[new_labeled_set[time]].view(-1).to(device)
            #y_hat = models[network_type](x, edge_index, edge_attr)[new_labeled_set[time]].view(-1).to(device)
            y = victim_model(x, edge_index, edge_attr).detach().view(-1).to(device)
            y_hat = models[network_type](x, edge_index, edge_attr).view(-1).to(device)
            rgcn_loss += criterion(y_hat, y)
    optimizer.zero_grad(0)
    rgcn_loss = rgcn_loss / (time + 1)
    rgcn_loss.backward()
    optimizer.step()

def objective_function(models, dataset, device):
    fidelity = 0
    fidelity2 = 0
    fidelity3 = 0
    fidelity4 = 0
    gcn_fidelity =0
    accuracy = 0
    accuracy2 = 0
    accuracy3 = 0
    accuracy4 = 0
    gcn_accuracy = 0
    total_time = 0
    for time, snapshot in enumerate(dataset):
        with torch.cuda.device(device=device):
            x = snapshot.x.to(device)
            '''
            y = snapshot.y
            y_labels = torch.argmax(y, axis=1).to(device)
            '''
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            victim_labels = torch.argmax(models['VICTIM'].to(device)(x, edge_index, edge_attr).detach(), dim=1).long().clone().to(device)
            rgcn_labels = torch.argmax(models['RGCN'].to(device)(x, edge_index, edge_attr).detach(), dim=1).long().clone().to(device)
            rgcn2_labels = torch.argmax(models['RGCN2'].to(device)(x, edge_index, edge_attr).detach(), dim=1).long().clone().to(device)
            rgcn3_labels = torch.argmax(models['RGCN3'].to(device)(x, edge_index, edge_attr).detach(), dim=1).long().clone().to(device)
            rgcn4_labels = torch.argmax(models['RGCN4'].to(device)(x, edge_index, edge_attr).detach(), dim=1).long().clone().to(device)
            gcn_labels = torch.argmax(models['GCN'].to(device)(x, edge_index, edge_attr).detach(), dim=1).long().clone().to(device)
            '''
            fidelity  += f1_score(victim_labels, rgcn_labels, average='binary')
            fidelity2 += f1_score(victim_labels, rgcn2_labels, average='binary')
            fidelity3 += f1_score(victim_labels, rgcn3_labels, average='binary')
            fidelity4 += f1_score(victim_labels, rgcn4_labels, average='binary')
            gcn_fidelity += f1_score(victim_labels, gcn_labels, average='binary')
            '''
            #print(victim_labels)
            #print(rgcn_labels)
            #print(rgcn2_labels)
            fidelity += torch.eq(victim_labels, rgcn_labels).sum() / victim_labels.shape[0]
            fidelity2 += torch.eq(victim_labels, rgcn2_labels).sum() / victim_labels.shape[0]
            fidelity3 += torch.eq(victim_labels, rgcn3_labels).sum() / victim_labels.shape[0]
            fidelity4 += torch.eq(victim_labels, rgcn4_labels).sum() / victim_labels.shape[0]
            gcn_fidelity += torch.eq(victim_labels, gcn_labels).sum() / victim_labels.shape[0]
            '''
            accuracy += torch.eq(y_labels, rgcn_labels).sum() / y_labels.shape[0]
            accuracy2 += torch.eq(y_labels, rgcn2_labels).sum() / y_labels.shape[0]
            accuracy3 += torch.eq(y_labels, rgcn3_labels).sum() / y_labels.shape[0]
            accuracy4 += torch.eq(y_labels, rgcn4_labels).sum() / y_labels.shape[0]
            gcn_accuracy += torch.eq(y_labels, gcn_labels).sum() / y_labels.shape[0]
            '''
            total_time += 1
    '''
    print('The fidelity result of rgcn is ' + str(fidelity / total_time))
    print('The fidelity result of rgcn2 is ' + str(fidelity2 / total_time))
    print('The fidelity result of rgcn3 is ' + str(fidelity3 / total_time))
    print('The fidelity result of rgcn4 is ' + str(fidelity4 / total_time))
    print('The fidelity result of gcn is ' + str(gcn_fidelity / total_time))
    print('The accuracy result of rgcn is ' + str(accuracy / total_time))
    print('The accuracy result of rgcn2 is ' + str(accuracy2 / total_time))
    print('The accuracy result of rgcn3 is ' + str(accuracy3 / total_time))
    print('The accuracy result of rgcn4 is ' + str(accuracy4 / total_time))
    print('The accuracy result of gcn is ' + str(gcn_accuracy / total_time))
    '''
    return fidelity / total_time, fidelity2 / total_time, fidelity3 / total_time, fidelity4 / total_time, gcn_fidelity / total_time


def objective_function2(models, dataset, device):
    #dataset = dataloaders['sim_train']
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
            victim_labels = models['VICTIM'](x, edge_index, edge_attr).clone().detach().to(device)
            rgcn_labels = models['RGCN'](x, edge_index, edge_attr).clone().detach().to(device)
            rgcn2_labels = models['RGCN2'](x, edge_index, edge_attr).clone().detach().to(device)
            rgcn3_labels = models['RGCN3'](x, edge_index, edge_attr).clone().detach().to(device)
            rgcn4_labels = models['RGCN4'](x, edge_index, edge_attr).clone().detach().to(device)
            gcn_labels = models['GCN'](x, edge_index, edge_attr).clone().detach().to(device)
            #print(victim_labels)
            #print(rgcn_labels)
            #print(rgcn4_labels)

            result += F.mse_loss(victim_labels, rgcn_labels)
            result2 += F.mse_loss(victim_labels, rgcn2_labels)
            result3 += F.mse_loss(victim_labels, rgcn3_labels)
            result4 += F.mse_loss(victim_labels, rgcn4_labels)
            gcn_result += F.mse_loss(victim_labels, gcn_labels)
            #print('1111111')
            #print(result)
            #print(result2)
            total_time += 1
    return result / total_time, result2 / total_time, result3 / total_time, result4 / total_time, gcn_result / total_time

