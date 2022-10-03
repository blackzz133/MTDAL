import math
import numpy as np
import torch
#from kmeans_pytorch import kmeans
from tqdm import tqdm
import networkx as nx
from collections import Counter
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.sparse import dense_to_sparse
from sklearn.cluster import KMeans

class Random_strategy:
    def __init__(self, dataset, models, labeled_set, cycle, all_loss, args, query_cost):
        self.dataset = dataset
        self.models = models
        self.labeled_set = labeled_set
        self.cycle = cycle
        self.budget = args.budget
        self.all_loss = all_loss
        self.time_span = torch.tensor(labeled_set).shape[0]
        self.node_num = torch.tensor(labeled_set).shape[1]
        self.total_cycle = round(self.time_span * args.budget / args.queries)
        #self.Rep= Rep #节点的representative
        #self.cluster_ids_x = cluster_ids_x
        self.class_num = args.class_num
        self.query_cost = query_cost


    def query(self, n):
        lset = torch.tensor(self.labeled_set)
        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()
        #torch.random.manual_seed(13)
        imp = torch.rand(self.time_span*self.node_num)
        unlabeled_imp = imp[~t_labeled_set]
        for i in tqdm(range(n), ncols=100):
            #返回各行的和
            l = -1
            #removed_ids = []
            #removed_imps = []
            while l == -1:
                q_idx_ = unlabeled_imp.argmax()
                #q_idx_ = unlabeled_imp.argmax()
                #获取最大的节点所在位置
                q_idx = np.arange(self.labeled_set.shape[0]*self.labeled_set.shape[1])[~t_labeled_set][q_idx_]
                start = q_idx//self.node_num* self.node_num
                end= start+self.node_num
                if len(np.ones(self.node_num)[t_labeled_set[start:end]].tolist()) >= self.budget:
                    print('111111')
                    #unlabeled_imp[q_idx_] = 0
                    #区间内所有imp值归0，以后不再选择
                    imp[start:end] = torch.tensor([-100 for i in range(start, end)])
                    unlabeled_imp = imp[~t_labeled_set]
                    #unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    continue
                else:
                    print('2222')
                    t_labeled_set[q_idx] = True
                    l = 0
                    unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
                    #每次选择多个
                    #unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    # unlabeled_imp中删除
        t_labeled_set = torch.tensor(t_labeled_set).reshape(self.time_span, self.node_num).numpy()
        return t_labeled_set
    #with the query cost
    def query2(self, n):
        #print('33')
        used_budget = 0
        lset = torch.tensor(self.labeled_set)
        t_query_cost = self.query_cost.reshape(lset.shape[0] * lset.shape[1]).numpy()
        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()
        #torch.random.manual_seed(7)
        qcost = t_query_cost[~t_labeled_set]
        qcost_min = np.array(qcost).min()
        imp = torch.rand(self.time_span * self.node_num)
        unlabeled_imp = imp[~t_labeled_set]
        for i in tqdm(range(n), ncols=100):
            if used_budget+qcost_min > n:
                break
            # 返回各行的和
            l = -1
            # removed_ids = []
            # removed_imps = []
            while l == -1:
                q_idx_ = unlabeled_imp.argmax()
                if unlabeled_imp[q_idx_] == -100:
                    break
                if used_budget + qcost[q_idx_] > n:
                    #imp[q_idx] = -100
                    #unlabeled_imp[q_idx_] = -100
                    unlabeled_imp[q_idx_] = -100
                    continue
                # 获取最大的节点所在位置
                q_idx = np.arange(self.labeled_set.shape[0] * self.labeled_set.shape[1])[~t_labeled_set][q_idx_]
                start = q_idx // self.node_num * self.node_num
                end = start + self.node_num
                _start = start - np.arange(start)[t_labeled_set[0:start]].shape[0]
                _end = end - np.arange(end)[t_labeled_set[0:end]].shape[0]
                #if len(np.ones(self.node_num)[t_labeled_set[start:end]].tolist()) >= self.budget \
                if sum(t_query_cost[start:end][t_labeled_set[start:end]]) + min(t_query_cost[start:end][~t_labeled_set[start:end]])> self.budget:
                    # unlabeled_imp[q_idx_] = 0
                    # 区间内所有imp值归0，以后不再选择
                    #imp[start:end] = torch.tensor([-100 for i in range(start, end)])
                    #unlabeled_imp = imp[~t_labeled_set]
                    unlabeled_imp[_start:_end] = torch.ones(_end-_start)*-100
                    #unlabeled_neighbor_imp[q_idx_] = -100
                    # unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    continue
                elif sum(t_query_cost[start:end][t_labeled_set[start:end]])+t_query_cost[q_idx]>self.budget:
                    unlabeled_imp[q_idx_] = -100
                    continue
                else:
                    t_labeled_set[q_idx] = True
                    l = 0
                    used_budget += t_query_cost[q_idx]
                    #unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
                    # 每次选择多个
                    unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
                    qcost = t_query_cost[~t_labeled_set]

                    # unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    # unlabeled_imp中删除
        t_labeled_set = torch.tensor(t_labeled_set).reshape(self.time_span, self.node_num).numpy()
        return t_labeled_set

