import torch
import numpy as np
import networkx as nx
from dataset_loader import ChickenpoxDatasetLoader, EnglandCovidDatasetLoader
from models import RecurrentGCN, ModelfreeGCN, GCN, A3TGCN_RecurrentGCN, DCRNN_RecurrentGCN, TGCN_RecurrentGCN, EvolveGCNO_RecurrentGCN
from TGCN.signal.train_test_split import temporal_signal_split,node_data_sampling2
from sklearn.metrics import f1_score
import torch.optim.lr_scheduler as lr_scheduler
from ModelExtraction.active_learning import *
import torch.nn.functional as F
from config import *
import copy as cp
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import networkx as nx
from torch_geometric.utils.convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import math

def Encovid_attack_model(args, victim_model, attack_type, device):
    loader1 = ChickenpoxDatasetLoader()
    loader2 = EnglandCovidDatasetLoader()
    sim_dataset = loader1.get_dataset(lags=4)
    raw_dataset = loader2.get_dataset(lags=4)
    sim_dataset = node_data_sampling2(sim_dataset, time_span=10)


    '''
    print(torch.tensor(sim_dataset.features).shape)
    print(torch.tensor(raw_dataset.features).shape)
    for time, snapshot in enumerate(sim_dataset):
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.cuda()
        edge_weight = snapshot.edge_attr.cuda()
        y_hat = victim_model(x,edge_index, edge_weight)
        print(y_hat)
        print('wwwwwwwwww')
        exit()
    exit()
    '''
    #sim_dataset = sim_dataset_generator(origin_dataset)  #需要完成
    #sim_dataset为攻击者具有的数据, raw dataset为拥有者的数据与ground truth
    sim_trainloader, sim_testloader = temporal_signal_split(sim_dataset, train_ratio=0.8)
    raw_trainloader, raw_testloader = temporal_signal_split(raw_dataset, train_ratio=0.8)  #提取
    victim_labels = []
    for time, snapshot in enumerate(sim_trainloader):
        x = snapshot.x.cuda()
        edge_index = snapshot.edge_index.cuda()
        edge_weight = snapshot.edge_attr.cuda()
        labels = torch.argmax(victim_model(x, edge_index, edge_weight).detach(), dim=1).long().tolist()
        victim_labels.append(labels)


    print(torch.tensor(sim_trainloader.features).shape)  #[5,6606,100]
    print(torch.tensor(raw_trainloader.features).shape)  #[12, 7278, 32]

    print(torch.tensor(sim_testloader.features).shape)  #[5,6606,100]
    print(torch.tensor(raw_testloader.features).shape)  #[12, 7278, 32]
    time_span = torch.tensor(sim_trainloader.features).shape[0]
    node_num = torch.tensor(sim_trainloader.features).shape[1]
    node_features = torch.tensor(sim_trainloader.features).shape[2]
    #num_classes = torch.tensor(raw_trainloader.targets).shape[2]
    # cycle访问次数,需要修改
    cycles = round(time_span * args.budget / args.queries)
    # print(time_span,node_num, node_features)
    # exit()
    labeled_set = np.array([np.zeros(node_num, dtype=bool) for i in range(time_span)])
    unlabeled_set = [~a for a in labeled_set]
    data_unlabeled = sim_trainloader

    # models
    model_free = ModelfreeGCN(node_features=node_features).to(device)
    model_rgcn = DCRNN_RecurrentGCN(node_features=node_features).to(device)
    model_rgcn2 = A3TGCN_RecurrentGCN(node_features=node_features).to(device)
    model_rgcn3 = EvolveGCNO_RecurrentGCN(node_features=node_features).to(device)
    model_rgcn4 = TGCN_RecurrentGCN(node_features=node_features).to(device)
    model_gcn = GCN(node_features=node_features).to(device)
    models = {'RGCN': model_rgcn, 'RGCN2': model_rgcn2, 'GCN': model_gcn, 'RGCN3': model_rgcn3, 'RGCN4': model_rgcn4,
              'FREE': model_free, 'VICTIM': victim_model}
    url = 'attack_models'
    if os.path.exists(url) == False:
        file = open(url, 'w')
        weights = {'RGCN': models['RGCN'].state_dict(), 'RGCN2': models['RGCN2'].state_dict(),
                   'RGCN3': models['RGCN3'].state_dict(), 'RGCN4': models['RGCN4'].state_dict(),
                   'GCN': models['GCN'].state_dict()}
        torch.save(weights, f=url)
    if os.path.getsize(url) > 0:
        print('Saved victim model is loaded')
        weights = torch.load(f=url)
        model_rgcn.load_state_dict(weights['RGCN'], strict=False)
        model_rgcn2.load_state_dict(weights['RGCN2'], strict=False)
        model_rgcn3.load_state_dict(weights['RGCN3'], strict=False)
        model_rgcn4.load_state_dict(weights['RGCN4'], strict=False)
        model_gcn.load_state_dict(weights['GCN'], strict=False)
        models = {'RGCN': model_rgcn, 'RGCN2': model_rgcn2, 'GCN': model_gcn, 'RGCN3': model_rgcn3,
                  'RGCN4': model_rgcn4, 'FREE': model_free, 'VICTIM': victim_model}

    # optimizers
    optimizer_rgcn = torch.optim.Adam(model_rgcn.parameters(), lr=0.001)
    optimizer_rgcn2 = torch.optim.Adam(model_rgcn2.parameters(), lr=0.001)
    optimizer_rgcn3 = torch.optim.Adam(model_rgcn3.parameters(), lr=0.001)
    optimizer_rgcn4 = torch.optim.Adam(model_rgcn4.parameters(), lr=0.001)
    optimizer_gcn = torch.optim.Adam(model_gcn.parameters(), lr=0.001)
    optimizers = {'optimizer_rgcn': optimizer_rgcn, 'optimizer_gcn': optimizer_gcn, 'optimizer_rgcn2': optimizer_rgcn2,
                  'optimizer_rgcn3': optimizer_rgcn3, 'optimizer_rgcn4': optimizer_rgcn4}
    # criterions
    criterion_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
    criterion_mse = torch.nn.MSELoss(reduction='mean')
    criterion_f1 = f1_score
    criterions = {'mse': criterion_mse, 'f1': criterion_f1, 'cross_entropy': criterion_cross_entropy}
    # schedulers
    scheduler_rgcn = lr_scheduler.StepLR(optimizer_rgcn, step_size=20, gamma=0.2)
    scheduler_gcn = lr_scheduler.StepLR(optimizer_gcn, step_size=20, gamma=0.2)
    scheduler_rgcn2 = lr_scheduler.StepLR(optimizer_rgcn2, step_size=20, gamma=0.2)
    scheduler_rgcn3 = lr_scheduler.StepLR(optimizer_rgcn3, step_size=20, gamma=0.2)
    scheduler_rgcn4 = lr_scheduler.StepLR(optimizer_rgcn4, step_size=20, gamma=0.2)
    schedulers = {'scheduler_rgcn': scheduler_rgcn, 'scheduler_gcn': scheduler_gcn, 'scheduler_rgcn2': scheduler_rgcn2,
                  'scheduler_rgcn3': scheduler_rgcn3, 'scheduler_rgcn4': scheduler_rgcn4}
    # 存储test loss
    all_loss = {'RGCN': 0, 'RGCN2': 0, 'GCN': 0, 'RGCN3': 0, 'RGCN4': 0}
    all_loss2 = {'RGCN': 0, 'RGCN2': 0, 'GCN': 0, 'RGCN3': 0, 'RGCN4': 0}
    all_loss3 = {'RGCN': 0, 'RGCN2': 0, 'GCN': 0, 'RGCN3': 0, 'RGCN4': 0}
    # device
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # dataloaders
    # sim_train:用来进行active learning, sim_test：用来进行结果的验证， raw_train源模型的训练，raw_test:active learning时候的ground truth
    dataloaders = {'sim_train': sim_trainloader, 'sim_test': sim_testloader, 'raw_train': raw_trainloader,
                   'raw_test': raw_testloader}
    print('Model extraction is started')
    query_url = 'query_cost.cost'
    if os.path.exists(query_url) == False:
        file = open(query_url, 'w')
    if os.path.getsize(query_url) > 0:
        print('Saved victim model is loaded')
        q_cost = torch.load(f=query_url)
    else:
        cost = np.zeros((time_span, node_num))
        q_cost = query_cost(cost, dataloaders['sim_train'])
    print('query cost is loaded')
    print(max(q_cost[0]), min(q_cost[0]))
    print(max(q_cost[1]), min(q_cost[1]))

    # query_test
    q_cost = torch.ones(q_cost.shape)
    method = 'first_query'
    print('The start ' + str(time_span) + ' cycles is started')
    all_loss['RGCN'], all_loss['RGCN2'], all_loss['RGCN3'], all_loss['RGCN4'], all_loss['GCN'] = objective_function2(
        models, raw_testloader, device)
    all_loss2['RGCN'], all_loss2['RGCN2'], all_loss2['RGCN3'], all_loss2['RGCN4'], all_loss2[
        'GCN'] = objective_function2(models, raw_trainloader, device)
    all_loss3['RGCN'], all_loss3['RGCN2'], all_loss3['RGCN3'], all_loss3['RGCN4'], all_loss3[
        'GCN'] = objective_function2(models, sim_trainloader, device)
    print('The initialization of all losses are :')
    print([float(i) for i in list(all_loss.values())])
    print([float(i) for i in list(all_loss2.values())])
    print([float(i) for i in list(all_loss3.values())])
    labeled_set = query_samples(models, method, dataloaders, labeled_set, 0, all_loss, args, q_cost)
    output_labeled_set(labeled_set, q_cost)
    all_loss = train2(models, criterions, optimizers, schedulers, dataloaders, labeled_set, 10, all_loss, device)
    all_loss['RGCN'], all_loss['RGCN2'], all_loss['RGCN3'], all_loss['RGCN4'], all_loss[
        'GCN'] = objective_function2(
        models, raw_testloader, device)
    all_loss2['RGCN'], all_loss2['RGCN2'], all_loss2['RGCN3'], all_loss2['RGCN4'], all_loss2[
        'GCN'] = objective_function2(models, raw_trainloader, device)
    all_loss3['RGCN'], all_loss3['RGCN2'], all_loss3['RGCN3'], all_loss3['RGCN4'], all_loss3[
        'GCN'] = objective_function2(models, sim_trainloader, device)
    print('The first ' + str(time_span) + ' cycles of active learning is finished')
    print([float(i) for i in list(all_loss.values())])
    print([float(i) for i in list(all_loss2.values())])
    print([float(i) for i in list(all_loss3.values())])

    for cycle in range(time_span, cycles + 1):

        print(cycle)
        first_cycle_epochs = args.no_of_epochs  # t1
        last_cycle_epochs = args.no_of_epochs * 2
        current_cycle_epochs = round((last_cycle_epochs - first_cycle_epochs) * cycle / cycles + first_cycle_epochs)
        if attack_type == 'Kcenter':
            labeled_set = query_samples(models, attack_type, dataloaders, labeled_set, cycle, all_loss, args, q_cost)

        if attack_type == 'ALTG':
            # new_models = cp.deepcopy(models)
            # new_models.pop('GCN')
            labeled_set = query_samples(models, attack_type, dataloaders, labeled_set, cycle, all_loss, args, q_cost,
                                        victim_labels)

            # print(np.arange(len(labeled_set[1]))[labeled_set[1]])

            # exit()
        if attack_type == 'DEAL':
            labeled_set = query_samples(models, attack_type, dataloaders, labeled_set, cycle, all_loss, args, q_cost)

        if attack_type == 'MTTAL':
            labeled_set = query_samples(models, attack_type, dataloaders, labeled_set, cycle, all_loss, args, q_cost,
                                        victim_labels)

        elif attack_type == 'Random':
            labeled_set = query_samples(models, attack_type, dataloaders, labeled_set, cycle, all_loss, args, q_cost)
        output_labeled_set(labeled_set, q_cost)
        all_loss = train2(models, criterions, optimizers, schedulers, dataloaders, labeled_set, current_cycle_epochs,
                          all_loss, device)

        # print(all_loss)
        # 输出目标函数
        print('All losses are: ')

        all_loss['RGCN'], all_loss['RGCN2'], all_loss['RGCN3'], all_loss['RGCN4'], all_loss[
            'GCN'] = objective_function2(
            models, raw_testloader, device)
        all_loss2['RGCN'], all_loss2['RGCN2'], all_loss2['RGCN3'], all_loss2['RGCN4'], all_loss2[
            'GCN'] = objective_function2(
            models, raw_trainloader, device)
        all_loss3['RGCN'], all_loss3['RGCN2'], all_loss3['RGCN3'], all_loss3['RGCN4'], all_loss3[
            'GCN'] = objective_function2(models, sim_trainloader, device)
        print([float(i) for i in list(all_loss.values())])
        print([float(i) for i in list(all_loss2.values())])
        print([float(i) for i in list(all_loss3.values())])
        # result, result2, result3, result4, gcn_result = objective_function(models, raw_trainloader, device)
        # all_loss = test(models, criterions, dataloaders, device, mode='test')
    return models, all_loss

#model is models['FREE']
def rep(model, dataloaders):
    dataset = dataloaders['sim_train']
    raw_dataset = dataloaders['raw_train']
    #转换list
    num_clusters = torch.tensor(raw_dataset.targets).shape[2]
    all_embeddings = []
    for time, snapshot in enumerate(dataset):
        embeddings = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).detach()
        embeddings = embeddings.tolist()
        all_embeddings.append(embeddings)
    all_embeddings = torch.tensor(all_embeddings)
    #print(self.labeled_set.shape[0],self.labeled_set.shape[1], self.labeled_set.shape[0]*self.labeled_set.shape[1])
    all_embeddings = all_embeddings.reshape(int(all_embeddings.shape[0] * all_embeddings.shape[1]),
                                            all_embeddings.shape[2])
    km = KMeans(n_clusters=num_clusters, random_state=0, max_iter=100).fit(all_embeddings)
    cluster_ids_x = torch.tensor(km.labels_)
    cluster_centers = torch.tensor(km.cluster_centers_)
    # cluster_ids_x, cluster_centers = kmeans(X=all_embeddings, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
    Rep = 1 - torch.abs(torch.tensor(
        [torch.dist(all_embeddings[i], cluster_centers[cluster_ids_x[i]]) for i in range(all_embeddings.shape[0])]))
    max_rep = [-100 for i in range(torch.tensor(raw_dataset.targets).shape[2])]

    min_rep = [100 for i in range(torch.tensor(raw_dataset.targets).shape[2])]
    for i in range(Rep.shape[0]):
        max_rep[cluster_ids_x[i]] = max(max_rep[cluster_ids_x[i]], Rep[i])
        min_rep[cluster_ids_x[i]] = min(min_rep[cluster_ids_x[i]], Rep[i])
    Rep = torch.tensor([(Rep[i]-min_rep[cluster_ids_x[i]])/(max_rep[cluster_ids_x[i]] - min_rep[cluster_ids_x[i]]) for j in range(Rep.shape[0])])
    return cluster_ids_x, Rep

def query_cost(q_cost, dataset):
    a = 3 #parameter of query cost
    b = 2
    layer = torch.nn.Sigmoid()
    new_q_cost = cp.deepcopy(q_cost)
    time_span = new_q_cost.shape[0]
    node_num = new_q_cost.shape[1]
    for time, snapshot in enumerate(dataset):
        x = snapshot.x
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr
        matrix = to_scipy_sparse_matrix(edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])
        graph = nx.Graph(incoming_graph_data= matrix)
        # centrality of each node
        #print(list(dict(nx.betweenness_centrality(graph)).values()))
        #exit()
        bet = nx.degree_centrality(graph)
        #bet = nx.betweenness_centrality(graph)
        new_q_cost[time] = np.array(list(dict(bet).values()))
    #print('xxx')
    new_q_cost2 = cp.deepcopy(new_q_cost)
    new_q_cost3 = cp.deepcopy(new_q_cost)
    for i in range(0, time_span):
        if i != 0:
            for j in range(node_num):
                new_q_cost2[i][j] = (new_q_cost[i - 1][j] + new_q_cost[i][j]) / 2
        max_cost = np.max(new_q_cost[time])
        min_cost = np.min(new_q_cost[time])
        #print("22222")
        #print(max_cost)
        #print(min_cost)
        # max query is 3, min query is 1
        new_q_cost2[i] = b*(new_q_cost2[i] - min_cost) / (max_cost - min_cost)
        new_q_cost2[i] = a*layer(torch.tensor(new_q_cost2[i].tolist())).numpy()
        #print(new_q_cost3[i])
    return torch.from_numpy(new_q_cost2)

# use centrality as query cost
def query_cost2(q_cost, dataset):
    a = 8 #parameter of query cost
    new_q_cost = cp.deepcopy(q_cost)
    time_span = new_q_cost.shape[0]
    node_num = new_q_cost.shape[1]
    for time, snapshot in enumerate(dataset):
        x = snapshot.x
        edge_index = snapshot.edge_index
        edge_attr = snapshot.edge_attr
        matrix = to_scipy_sparse_matrix(edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])
        graph = nx.Graph(incoming_graph_data= matrix)
        # centrality of each node
        #print(list(dict(nx.betweenness_centrality(graph)).values()))
        #exit()
        bet = nx.degree_centrality(graph)
        #bet = nx.betweenness_centrality(graph)
        new_q_cost[time] = np.array(list(dict(bet).values()))
    #print('xxx')
    new_q_cost2 = cp.deepcopy(new_q_cost)
    new_q_cost3 = cp.deepcopy(new_q_cost)
    for i in range(0, time_span):
        if i != 0:
            for j in range(node_num):
                new_q_cost2[i][j] = (new_q_cost[i - 1][j] + new_q_cost[i][j]) / 2
        max_cost = np.max(new_q_cost2[time])
        min_cost = np.min(new_q_cost2[time])
        #print("22222")
        #print(max_cost)
        #print(min_cost)
        # max query is 3, min query is 1
        new_q_cost3[i] = (new_q_cost2[i] - min_cost) * a / (max_cost - min_cost) + 1
        #print(new_q_cost3[i])
    return torch.from_numpy(new_q_cost3)



def output_labeled_set(labeled_set,q_cost):
    b = np.ones(labeled_set.shape)
    c = [len(b[i][labeled_set[i]].tolist()) for i in range(b.shape[0])]
    d = [float(sum(q_cost[i][labeled_set[i]])) for i in range(b.shape[0])]
    #print(np.arange(len(labeled_set[0]))[labeled_set[0]])
    print(c)
    print(sum(c))
    print(d)
    print(sum(d))
    print([np.arange(b.shape[1])[labeled_set[i]].tolist() for i in range(b.shape[0])])
