import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from kmeans_pytorch import kmeans
#from data.sampler import SubsetSequentialSampler
#from ModelExtraction.active_learning.query_strategies.adversarial_deepfool import DeepFool
from ModelExtraction.active_learning import query_strategies
from torch_geometric.nn import GCNConv
from ModelExtraction.active_learning.query_strategies import KCenterGreedy
from ModelExtraction.active_learning.query_strategies.active_learning_temporal_graph import ALTG_strategy
from ModelExtraction.active_learning.query_strategies.tgcn_active_learning import MTTAL_strategy
from ModelExtraction.active_learning.query_strategies.random import Random_strategy
#from ModelExtraction.active_learning.query_strategies.active_learning_graph import ALG
import copy as cp

#a= data.sampler
def query_samples(models, method, dataloaders, labeled_set, cycle, all_loss, args, query_cost=None, victim_labels = None):
    #time_span = len(labeled_set)
    #kcenter, time-sensitive kcenter, DEAL+kcenter, ALG,
    if method == 'Kcenter+GCN':
        models.pop('GCN')
        arg = Kcenter(models, dataloaders, labeled_set, args)
    elif method == 'Kcenter+RGCN':
        models.pop('RGCN')
        arg = Kcenter(models, dataloaders, labeled_set,  args)
    elif method == 'Kcenter':
        arg = Kcenter(models, dataloaders, labeled_set, all_loss, args, query_cost)
    elif method == 'first_query':
            arg = first_query_samples2(models, dataloaders, labeled_set, args, query_cost)
    elif method == 'ALG':
        arg = ALTG(models, dataloaders, labeled_set, cycle, all_loss, args)
    elif method == 'ALTG':
        arg = ALTG(models, dataloaders, labeled_set, cycle, all_loss, args, query_cost, victim_labels)
    elif method =='DEAL+Kcenter':
        arg = first_query_samples(models, dataloaders, labeled_set, args)
    elif method =='ALTG2':  #for test
        arg = ALTG2(models, dataloaders, labeled_set, cycle, all_loss, args, query_cost, victim_labels)
    elif method =='MTTAL':
        arg = MTTAL(models, dataloaders, labeled_set, cycle, all_loss, args, query_cost, victim_labels)
    elif method =='Random':
        arg = Random(models, dataloaders, labeled_set, cycle, all_loss, args, query_cost)
    elif method =='M_Kcenter':
        arg = M_Kcenter(models, dataloaders, labeled_set, cycle, all_loss, args, query_cost, victim_labels)
    return arg



#不按照query_cost选择节点
def first_query_samples(models, dataloaders, labeled_set, args, query_cost):
    dataset = dataloaders['sim_train']
    model = models['FREE']
    query_num = args.queries
    for time, snapshot in enumerate(dataset):
        x = snapshot.x
        n_pool = x.shape[0]
        edge_index = snapshot.edge_index
        edge_weight = snapshot.edge_attr
        labeled_idxs = np.zeros(len(x), dtype=bool)
        embeddings = model(x, edge_index, edge_weight)
        embeddings = embeddings.detach().numpy()
        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        #print(sq)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(np.abs(dist_mat))
        mat = dist_mat[~labeled_idxs, :][:, ~labeled_idxs]
        #nonzero_mat = cp.deepcopy(mat)
        #把到自身的距离设置为最大
        for i in range(mat.shape[0]):
            mat[i][i] = 10000
        for i in tqdm(range(query_num), ncols=100):
            # 返回各行的最小值
            mat_min = mat.min(axis=1)
            # 从未标注的各行和中选出最大的
            q_idx_ = mat_min.argmax()
            #print('####')
            #print(mat_sum[q_idx_])
            #print('1111111111')
            #exit()
            # 获取最大的节点所在位置
            q_idx = np.arange(n_pool)[~labeled_idxs][q_idx_]
            # 标注为True
            labeled_idxs[q_idx] = True
            # mat中删除
            #print('xxxx')
            #print(mat.shape) #(6606, 6606)
            mat = np.delete(mat, q_idx_, 0)
            #print(mat.shape) #(6605, 6606)
            #增加新加入的q_idx
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
            #print(mat.shape) #(6605, 6607)
            #print('yyy')
            #exit()
            #break
        labeled_set[time] = labeled_idxs
    return np.array(labeled_set)

#选择第一轮的samples
def first_query_samples2(models, dataloaders, labeled_set, args, query_cost):
    dataset = dataloaders['sim_train']
    model = models['FREE']
    query_num = args.queries
    for time, snapshot in enumerate(dataset):
        x = snapshot.x
        n_pool = x.shape[0]
        edge_index = snapshot.edge_index
        edge_weight = snapshot.edge_attr
        labeled_idxs = np.zeros(len(x), dtype=bool)
        embeddings = model(x, edge_index, edge_weight)
        embeddings = embeddings.detach().numpy()
        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        #print(sq)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(np.abs(dist_mat))
        mat = dist_mat[~labeled_idxs, :][:, ~labeled_idxs]
        qcost = query_cost[time].numpy()[~labeled_idxs]
        min_qcost = qcost.min(axis=0)
        #nonzero_mat = cp.deepcopy(mat)
        #把到自身的距离设置为最大
        for i in range(mat.shape[0]):
            mat[i][i] = 10000
        for j in tqdm(range(query_num), ncols=100):
            used_budget = sum(query_cost[time][labeled_idxs])
            #print('used_budget')
            #print(used_budget)
           #超出了预算，结束该轮
            if used_budget+min_qcost > query_num:
                labeled_set[time] = labeled_idxs
                #print('111111111111111111')
                break
            # 返回各行的最小值
            mat_min = mat.min(axis=1)
            #保证所选择的每行均不会大于预算
            for i in range(mat_min.shape[0]):
                if used_budget + qcost[i] > query_num:
                    mat_min[i] = 0
                else:
                    mat_min[i] = mat_min[i] / qcost[i]
            # 从未标注的各行和中选出最大的
            q_idx_ = mat_min.argmax()
            #print('$$$$')
            #print(mat_min[q_idx_])
            #print(qcost[q_idx_])
            #print('####')
            #print(mat_sum[q_idx_])
            #print('1111111111')
            #exit()
            # 获取最大的节点所在位置
            q_idx = np.arange(n_pool)[~labeled_idxs][q_idx_]
            # 标注为True
            #代价超出了本次的预算query_num
            if used_budget+qcost[q_idx_]>query_num:
                mat[q_idx] = np.zeros(mat[q_idx].shape)
                continue
            labeled_idxs[q_idx] = True

            # mat中删除
            #print('xxxx')
            #print(mat.shape) #(6606, 6606)
            mat = np.delete(mat, q_idx_, 0)
            #print(mat.shape) #(6605, 6606)
            #增加新加入的q_idx
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
            qcost = query_cost[time].numpy()[~labeled_idxs]
            #break
            #print(mat.shape) #(6605, 6607)
            #print('yyy')
            #exit()
        labeled_set[time] = labeled_idxs
    return np.array(labeled_set)

def Kcenter(models, dataloaders, labeled_set, all_loss, args, query_cost):
    KCG = KCenterGreedy(dataloaders['sim_train'], models, labeled_set, args.budget, query_cost)
    res = max(all_loss, key=lambda x: all_loss[x])
    if query_cost == None:
        arg = KCG.query(args.queries, res)
    else:
        arg = KCG.query2(args.queries, res)
    # 不能超过budget
    return np.array(arg)

def M_Kcenter(models, dataloaders, labeled_set, cycle, all_loss, args, query_cost, victim_labels):
    dataset = dataloaders['sim_train']
    KCG = KCenterGreedy(dataloaders['sim_train'], models, labeled_set, args.budget, query_cost)
    res = max(all_loss, key=lambda x: all_loss[x])
    #select nodes from mutiple times kcenter result
    mutiple = 2
    selected_ids = KCG.query3(args.queries*mutiple, res)
    selected_ids = list(set(selected_ids))
    print(selected_ids)
    print('Kcenter Result is generated')
    M = MTTAL_strategy(dataset, models, labeled_set, cycle, all_loss, args, query_cost, victim_labels, selected_ids)

    arg = M.query3(args.queries)
    # 不能超过budget
    return np.array(arg)




def ALTG(models, dataloaders, labeled_set, cycle, all_loss, args, query_cost, victim_labels):
    dataset = dataloaders['sim_train']
    A = ALTG_strategy(dataset, models, labeled_set, cycle, all_loss, args, query_cost, victim_labels)
    #A = ALTG_strategy(dataset, models, labeled_set, cycle, all_loss, args, Rep, cluster_ids_x)
    arg = A.query2(args.queries)
    #dataset = dataloaders['sim_train']
    #time_span = torch.tensor(dataset.features).shape[0]
    #num_clusters = torch.tensor(dataset.targets).shape[2]
    #cycles = round(time_span * args.budget / args.queries)
    return np.array(arg)

def ALTG2(models, dataloaders, labeled_set, cycle, all_loss, args, query_cost, victim_labels):
    dataset = dataloaders['sim_train']
    A = ALTG_strategy(dataset, models, labeled_set, cycle, all_loss, args, query_cost, victim_labels)
    #A = ALTG_strategy(dataset, models, labeled_set, cycle, all_loss, args, Rep, cluster_ids_x)
    arg = A.query3(args.queries)
    #dataset = dataloaders['sim_train']
    #time_span = torch.tensor(dataset.features).shape[0]
    #num_clusters = torch.tensor(dataset.targets).shape[2]
    #cycles = round(time_span * args.budget / args.queries)
    return np.array(arg)



def MTTAL(models, dataloaders, labeled_set, cycle, all_loss, args, query_cost, victim_labels):
    dataset = dataloaders['sim_train']
    M = MTTAL_strategy(dataset, models, labeled_set, cycle, all_loss, args, query_cost, victim_labels)
    # A = ALTG_strategy(dataset, models, labeled_set, cycle, all_loss, args, Rep, cluster_ids_x)
    arg = M.query2(args.queries)


    # dataset = dataloaders['sim_train']
    # time_span = torch.tensor(dataset.features).shape[0]
    # num_clusters = torch.tensor(dataset.targets).shape[2]
    # cycles = round(time_span * args.budget / args.queries)
    return np.array(arg)

def Random(models, dataloaders, labeled_set, cycle, all_loss, args, query_cost):
    dataset = dataloaders['sim_train']
    R = Random_strategy(dataset, models, labeled_set, cycle, all_loss, args, query_cost)
    arg = R.query2(args.queries)
    return np.array(arg)