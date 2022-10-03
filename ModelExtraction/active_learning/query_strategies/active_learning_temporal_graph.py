import math
import numpy as np
import torch
#from kmeans_pytorch import kmeans
from tqdm import tqdm
import networkx as nx
from collections import Counter
import copy as cp
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.sparse import dense_to_sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from ModelExtraction.active_learning.utils import SupervisedKMeans, SemiKMeans

class ALTG_strategy:
    def __init__(self, dataset, models, labeled_set, cycle, all_loss, args, query_cost, victim_labels):
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
        self.victim_labels = victim_labels
        self.class_num = args.class_num
        self.query_cost = query_cost


    def query(self, n):

        lset = torch.tensor(self.labeled_set)
        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()
        alph = math.cos(math.pi*(self.cycle+1)/(2*self.total_cycle))
        beta = 0
        # imp: importance of nodes
        # 有问题
        #a = self.rep()
        #cluster_ids_x, Rep = self.rep(self.models['FREE'], self.dataset)
        a = self.Rep
        b = self.inf()###################################有问题
        #print('wwwwww')
        #print(a)
        #print(b)
        imp = alph*a+(1-alph)*b

        unlabeled_imp = imp[~t_labeled_set]
        edge_indices = self.dataset.edge_indices
        neighbor_dics = {}
        for i in range(self.time_span):
            graph = nx.Graph()
            graph.add_nodes_from([j for j in range(self.node_num)])
            edges= torch.tensor(edge_indices[i]).t().tolist()
            graph.add_edges_from(edges)
            neighbor_dic = {j: graph.neighbors(j) for j in range(self.node_num)}
            neighbor_dics[i] = neighbor_dic
        unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
        #exit()
        count = Counter(self.cluster_ids_x[t_labeled_set].tolist())

        # constraint of selected node
        #print(len(np.arange(self.time_span*self.node_num)[t_labeled_set].tolist())+n)
        c = (len(np.arange(self.time_span*self.node_num)[t_labeled_set].tolist())+n)/self.class_num
        for i in tqdm(range(n), ncols=100):

            #返回各行的和
            l = -1
            #removed_ids = []
            #removed_imps = []
            while l == -1:
                q_idx_ = unlabeled_neighbor_imp.argmax()

                #q_idx_ = unlabeled_imp.argmax()
                #获取最大的节点所在位置
                q_idx = np.arange(self.labeled_set.shape[0]*self.labeled_set.shape[1])[~t_labeled_set][q_idx_]
                class_q = self.cluster_ids_x[q_idx]
                #某类数据选择后,该类数据的数量如果大于c
                if count[class_q]+1 > c:
                    #暂时存储不选的
                    #removed_ids.append(q_idx_)
                    #removed_imps.append(unlabeled_neighbor_imp[q_idx_])
                    unlabeled_neighbor_imp[q_idx_] = -100
                    continue
                start = q_idx//self.node_num* self.node_num
                end= start+self.node_num
                if len(np.ones(self.node_num)[t_labeled_set[start:end]].tolist()) >= self.budget:
                    #unlabeled_imp[q_idx_] = 0
                    #区间内所有imp值归0，以后不再选择
                    imp[start:end] = torch.tensor([-100 for i in range(start, end)])
                    unlabeled_imp = imp[~t_labeled_set]
                    unlabeled_neighbor_imp[q_idx_] = -100
                    #unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    continue
                else:
                    t_labeled_set[q_idx] = True
                    l = 0
                    count[class_q] += 1
                    unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
                    #每次选择多个
                    unlabeled_neighbor_imp = np.delete(unlabeled_neighbor_imp, q_idx_, 0)

                    #unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    # unlabeled_imp中删除
        t_labeled_set = torch.tensor(t_labeled_set).reshape(self.time_span, self.node_num).numpy()
        return t_labeled_set
    #with the query cost
    def query2(self, n):
        #determines which rgcn better objective
        self.all_loss
        res = max(self.all_loss, key=lambda x: self.all_loss[x])
        #test
        #res = 'RGCN'
        model = cp.deepcopy(self.models[res])
        used_budget = 0
        lset = torch.tensor(self.labeled_set)
        t_query_cost = self.query_cost.reshape(lset.shape[0] * lset.shape[1]).numpy()
        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()
        qcost = t_query_cost[~t_labeled_set]
        qcost_min = np.array(qcost).min()

        alph = math.cos(math.pi * (self.cycle + 1) / (2 * self.total_cycle))
        beta = 0
        # imp: importance of nodes
        # 有问题
        # a = self.rep()
        # cluster_ids_x, Rep = self.rep(self.models['FREE'], self.dataset)
        #representative
        cluster_ids_x, a = self.rep(self.models['FREE'], self.dataset)
        #print(a)
        #exit()
        b = self.inf()  ###################################有问题
        # print('wwwwww')
        # print(a)
        # print(b)
        imp = alph * a + (1 - alph) * b

        unlabeled_imp = imp[~t_labeled_set]
        edge_indices = self.dataset.edge_indices
        neighbor_dics = {}
        for i in range(self.time_span):
            graph = nx.Graph()
            graph.add_nodes_from([j for j in range(self.node_num)])
            edges = torch.tensor(edge_indices[i]).t().tolist()
            graph.add_edges_from(edges)
            neighbor_dic = {j: graph.neighbors(j) for j in range(self.node_num)}
            neighbor_dics[i] = neighbor_dic
        unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)

        # exit()
        #count = Counter(cluster_ids_x[t_labeled_set].tolist())
        #获取每个类上消耗的预算
        t_victim_labels = torch.tensor(self.victim_labels).reshape(lset.shape[0]*lset.shape[1])[t_labeled_set]
        count = {i:0 for i in range(self.class_num)}
        for i in range(cluster_ids_x[t_labeled_set].shape[0]):
            #count[cluster_ids_x[i]] += t_query_cost[t_labeled_set][i]
            #print(self.victim_labels[t_labeled_set])
            #print(self.victim_labels[t_labeled_set][i])
            count[int(t_victim_labels[i])]+= t_query_cost[t_labeled_set][i]
        print('www')
        print(count)
        #cluster_ids_x[t_labeled_set]

        # constraint of selected node
        #c = (len(np.arange(self.time_span * self.node_num)[t_labeled_set].tolist()) + n) / self.class_num
        c = n*(self.cycle+1)/self.class_num
        print('the c is '+str(c))
        for i in tqdm(range(n), ncols=100):
            if used_budget+qcost_min > n:
                break
            # 返回各行的和
            l = -1
            # removed_ids = []
            # removed_imps = []
            while l == -1:
                q_idx_ = (unlabeled_neighbor_imp/(qcost**1.5)).argmax()
                #-100会传到下一轮，-200只保证本轮，轮次更换标志为给数据标签
                if unlabeled_neighbor_imp[q_idx_] == -100 or unlabeled_neighbor_imp[q_idx_] == -200:
                    break
                if used_budget + qcost[q_idx_] > n:
                    #imp[q_idx] = -100
                    #unlabeled_imp[q_idx_] = -100
                    unlabeled_neighbor_imp[q_idx_] = -200
                    continue
                # 获取最大的节点所在位置
                q_idx = np.arange(self.labeled_set.shape[0] * self.labeled_set.shape[1])[~t_labeled_set][q_idx_]
                # class_q需要进行修改

                time = q_idx//self.node_num
                feature = torch.tensor(self.dataset.features[time],dtype=torch.float).cuda()
                edge_indice = torch.tensor(self.dataset.edge_indices[time],dtype=torch.long).cuda()
                edge_weight = torch.tensor(self.dataset.edge_weights[time],dtype=torch.float).cuda()
                out = model(feature, edge_indice, edge_weight).clone().detach()
                class_q = int(torch.argmax(out,dim=1).long()[q_idx%self.node_num])

                #print('cc')
                #print(class_q)
                #print(count[class_q])
                #print(qcost)
                #print(c)
                #class_q = cluster_ids_x[q_idx]
                # 某类数据选择后,该类数据的数量如果大于c
                #if count[class_q] + 1 > c:
                if count[class_q]+qcost[q_idx_] >c:
                    # 暂时存储不选的
                    unlabeled_neighbor_imp[q_idx_] = -200
                    continue
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
                    unlabeled_neighbor_imp[_start:_end] = torch.ones(_end-_start)*-100
                    #unlabeled_neighbor_imp[q_idx_] = -100
                    # unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    continue
                elif sum(t_query_cost[start:end][t_labeled_set[start:end]])+t_query_cost[q_idx]>self.budget:
                    unlabeled_neighbor_imp[q_idx_] = -100
                    continue
                else:
                    print("the pre-class is "+str(class_q))
                    #print(q_idx_)
                    #print(unlabeled_neighbor_imp[q_idx_])
                    t_labeled_set[q_idx] = True
                    #self.victim_labels
                    #c = (len(np.arange(self.time_span * self.node_num)[t_labeled_set].tolist()) + n) / self.class_num
                    l = 0
                    count[class_q] += qcost[q_idx_]
                    #print('33333')
                    #print(class_q)
                    used_budget += t_query_cost[q_idx]
                    #unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
                    # 每次选择多个
                    unlabeled_neighbor_imp = np.delete(unlabeled_neighbor_imp, q_idx_, 0)
                    qcost = t_query_cost[~t_labeled_set]
                    #重制imp
                    #print('0')
                    unlabeled_neighbor_imp = self.neighbor_add_imp_in_time(t_labeled_set, neighbor_dics, unlabeled_neighbor_imp, imp, q_idx//self.node_num)
                    #print('1')
                    # unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    # unlabeled_imp中删除
        t_labeled_set = torch.tensor(t_labeled_set).reshape(self.time_span, self.node_num).numpy()
        '''
        for j in range(self.labeled_set.shape[0]):
            c = np.ones(self.labeled_set[j].shape)
            print(len(c[self.labeled_set[j]]))
            print(len(c[t_labeled_set[j]]))
            self.labeled_set[j] = self.labeled_set[j] ^ t_labeled_set[j]
            print(len(c[self.labeled_set[j]]))
        '''
        return t_labeled_set


        #std = np.std(labeled_num_list, ddof=1)
    #一维和二维已经融合
    '''
    def rep(self):
        #转换list
        num_clusters = torch.tensor(self.dataset.targets).shape[2]
        all_embeddings = []
        for time, snapshot in enumerate(self.dataset):
            embeddings = self.models['FREE'](snapshot.x, snapshot.edge_index, snapshot.edge_attr).detach()
            embeddings = embeddings.tolist()
            all_embeddings.append(embeddings)
        all_embeddings = torch.tensor(all_embeddings)
        #print(self.labeled_set.shape[0],self.labeled_set.shape[1], self.labeled_set.shape[0]*self.labeled_set.shape[1])
        all_embeddings = all_embeddings.reshape(int(self.labeled_set.shape[0]*self.labeled_set.shape[1]), all_embeddings.shape[2])
        cluster_ids_x, cluster_centers = kmeans(
            X=all_embeddings, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
        )
        Rep = 1-torch.abs(torch.tensor([torch.dist(all_embeddings[i], cluster_centers[cluster_ids_x[i]]) for i in range(all_embeddings.shape[0])]))


        return Rep
    '''
    # without imp neighours
    def query3(self,n):
        used_budget = 0
        lset = torch.tensor(self.labeled_set)
        t_query_cost = self.query_cost.reshape(lset.shape[0] * lset.shape[1]).numpy()
        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()
        qcost = t_query_cost[~t_labeled_set]
        qcost_min = np.array(qcost).min()
        alph = math.cos(math.pi * (self.cycle + 1) / (2 * self.total_cycle))
        beta = 0
        # imp: importance of nodes
        # 有问题
        # a = self.rep()
        # cluster_ids_x, Rep = self.rep(self.models['FREE'], self.dataset)
        # representative
        cluster_ids_x, a = self.rep(self.models['FREE'], self.dataset)
        # print(a)
        # exit()
        b = self.inf()  ###################################有问题
        # print('wwwwww')
        # print(a)
        # print(b)
        imp = alph * a + (1 - alph) * b
        unlabeled_imp = imp[~t_labeled_set]
        edge_indices = self.dataset.edge_indices
        #unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)


        # exit()
        # count = Counter(cluster_ids_x[t_labeled_set].tolist())
        # 获取每个类上消耗的预算
        count = {i: 0 for i in range(self.class_num)}
        for i in range(cluster_ids_x[t_labeled_set].shape[0]):
            count[cluster_ids_x[i]] += t_query_cost[t_labeled_set][i]
        # cluster_ids_x[t_labeled_set]

        # constraint of selected node
        # c = (len(np.arange(self.time_span * self.node_num)[t_labeled_set].tolist()) + n) / self.class_num
        c = n * (self.cycle + 1) / self.class_num
        for i in tqdm(range(n), ncols=100):
            if used_budget + qcost_min > n:
                break
            # 返回各行的和
            l = -1
            # removed_ids = []
            # removed_imps = []
            while l == -1:
                q_idx_ = (unlabeled_imp / qcost).argmax()
                # -100会传到下一轮，-200只保证本轮，轮次更换标志为给数据标签
                if unlabeled_imp[q_idx_] == -100 or unlabeled_imp[q_idx_] == -200:
                    break
                if used_budget + qcost[q_idx_] > n:
                    # imp[q_idx] = -100
                    # unlabeled_imp[q_idx_] = -100
                    unlabeled_imp[q_idx_] = -200
                    continue
                # 获取最大的节点所在位置
                q_idx = np.arange(self.labeled_set.shape[0] * self.labeled_set.shape[1])[~t_labeled_set][q_idx_]
                class_q = cluster_ids_x[q_idx]
                # 某类数据选择后,该类数据的数量如果大于c
                # if count[class_q] + 1 > c:
                if count[class_q] + qcost[q_idx_] > c:
                    # 暂时存储不选的
                    unlabeled_imp[q_idx_] = -200
                    continue
                start = q_idx // self.node_num * self.node_num
                end = start + self.node_num
                _start = start - np.arange(start)[t_labeled_set[0:start]].shape[0]
                _end = end - np.arange(end)[t_labeled_set[0:end]].shape[0]
                # if len(np.ones(self.node_num)[t_labeled_set[start:end]].tolist()) >= self.budget \
                if sum(t_query_cost[start:end][t_labeled_set[start:end]]) + min(
                        t_query_cost[start:end][~t_labeled_set[start:end]]) > self.budget:
                    # unlabeled_imp[q_idx_] = 0
                    # 区间内所有imp值归0，以后不再选择
                    # imp[start:end] = torch.tensor([-100 for i in range(start, end)])
                    # unlabeled_imp = imp[~t_labeled_set]
                    unlabeled_imp[_start:_end] = torch.ones(_end - _start) * -100
                    # unlabeled_neighbor_imp[q_idx_] = -100
                    # unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    continue
                elif sum(t_query_cost[start:end][t_labeled_set[start:end]]) + t_query_cost[q_idx] > self.budget:
                    unlabeled_imp[q_idx_] = -100
                    continue
                else:
                    # print(q_idx_)
                    # print(unlabeled_neighbor_imp[q_idx_])
                    t_labeled_set[q_idx] = True
                    # c = (len(np.arange(self.time_span * self.node_num)[t_labeled_set].tolist()) + n) / self.class_num
                    l = 0
                    count[class_q] += qcost[q_idx_]
                    used_budget += t_query_cost[q_idx]
                    # unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
                    # 每次选择多个
                    unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
                    qcost = t_query_cost[~t_labeled_set]
                    # 重制imp
                    # print('0')

                    # print('1')
                    # unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    # unlabeled_imp中删除
        t_labeled_set = torch.tensor(t_labeled_set).reshape(self.time_span, self.node_num).numpy()
        '''
        for j in range(self.labeled_set.shape[0]):
            c = np.ones(self.labeled_set[j].shape)
            print(len(c[self.labeled_set[j]]))
            print(len(c[t_labeled_set[j]]))
            self.labeled_set[j] = self.labeled_set[j] ^ t_labeled_set[j]
            print(len(c[self.labeled_set[j]]))
        '''
        return t_labeled_set

    def inf(self):
        all_type_embeddings = {}
        INF = []
        for key, value in self.models.items():
            all_embeddings = []
            if (key in ['FREE','VICTIM'])==False:
            #if key!='FREE' and key!='victim' :
                for time, snapshot in enumerate(self.dataset):
                    model = self.models[key].cuda()
                    x= snapshot.x.cuda()
                    edge_index = snapshot.edge_index.cuda()
                    edge_attr = snapshot.edge_attr.cuda()
                    embeddings = model(x, edge_index, edge_attr).detach()
                    #print(embeddings)
                    embeddings = embeddings.tolist()
                    all_embeddings.append(embeddings)
                all_embeddings = torch.tensor(all_embeddings)
                all_embeddings = all_embeddings.reshape(self.labeled_set.shape[0] * self.labeled_set.shape[1],
                                                        all_embeddings.shape[2])
                all_type_embeddings[key] = all_embeddings
        '''
        print(all_type_embeddings['RGCN'])
        print(all_type_embeddings['RGCN2'])
        print(all_type_embeddings['RGCN3'])
        print(all_type_embeddings['RGCN4'])
        print(all_type_embeddings['GCN'])
        print('wwwwwwwwwwwwww')
        '''
        #exit()
        for i in range(self.labeled_set.shape[0]*self.labeled_set.shape[1]):
            #print('ccc')
            result = torch.tensor(0,dtype=float).cuda()
            for key, value in self.all_loss.items():
                for key2, value2 in self.all_loss.items():
                    error1 = 0.5 * torch.log2(value/(1-value)).cuda()
                    error2 = 0.5 * torch.log2(value2/(1-value2)).cuda()
                    result += error1*error2*torch.dist(all_type_embeddings[key][i], all_type_embeddings[key2][i])
            INF.append(float(result))

        INF = [(INF[i]-min(INF))/(max(INF)- min(INF)) for i in range(len(INF))]

        return torch.tensor(INF)

    def rep(self, model, dataset):
        #dataset = dataloaders['sim_train']
        #raw_dataset = dataloaders['raw_train']
        lset = torch.tensor(self.labeled_set)
        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()
        # 转换list
        num_clusters = self.class_num
        all_embeddings = []
        for time, snapshot in enumerate(dataset):
            embeddings = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).detach()
            embeddings = embeddings.tolist()
            all_embeddings.append(embeddings)
        all_embeddings = torch.tensor(all_embeddings)

        # print(self.labeled_set.shape[0],self.labeled_set.shape[1], self.labeled_set.shape[0]*self.labeled_set.shape[1])
        all_embeddings = all_embeddings.reshape(int(all_embeddings.shape[0] * all_embeddings.shape[1]),
                                                all_embeddings.shape[2])
        #semi-kmeans for getting the representative center
        y0 = np.array(self.victim_labels).reshape(self.time_span*self.node_num)[t_labeled_set]
        x0 = all_embeddings[t_labeled_set].numpy()
        t0 = np.arange(self.time_span*self.node_num)[t_labeled_set].tolist()
        t1 = np.arange(self.time_span*self.node_num)[~t_labeled_set].tolist()
        x1 = all_embeddings[~t_labeled_set].numpy()
        km = SemiKMeans(n_clusters=num_clusters)
        km.fit(x0, y0, x1)
        y1 = km.predict(x1)
        cluster_ids_x = np.ones(self.time_span*self.node_num,dtype=int)
        cluster_ids_x[t0] = y0
        cluster_ids_x[t1] = y1
        cluster_ids_x =np.array([int(i) for i in cluster_ids_x])
        #cluster_ids_x = torch.from_numpy(clusteridsx)
        #km = KMeans(n_clusters=num_clusters, random_state=0, max_iter=100).fit(all_embeddings)
        #cluster_ids_x = torch.tensor(km.labels_)
        cluster_centers = torch.tensor(km.cluster_centers_)
        # cluster_ids_x, cluster_centers = kmeans(X=all_embeddings, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'))
        ######

        Rep = 1 - torch.abs(torch.tensor(
            [torch.dist(all_embeddings[i], cluster_centers[cluster_ids_x[i]]) for i in range(all_embeddings.shape[0])]))
        max_rep = [-100 for i in range(self.class_num)]
        min_rep = [100 for i in range(self.class_num)]
        for i in range(Rep.shape[0]):
            max_rep[cluster_ids_x[i]] = max(max_rep[cluster_ids_x[i]], Rep[i])
            min_rep[cluster_ids_x[i]] = min(min_rep[cluster_ids_x[i]], Rep[i])
        Rep = torch.tensor(
            [(Rep[i] - min_rep[cluster_ids_x[i]]) / (max_rep[cluster_ids_x[i]] - min_rep[cluster_ids_x[i]]) for i in
             range(Rep.shape[0])])
        return cluster_ids_x, Rep
    #获取未标记节点带节点信息的imp
    def neighbor_add_imp(self, t_labeled_set, neighbor_dics, imp):
        #neighbor_dic 根据时间，节点获取邻居
        #print(neighbor_dics)
        add_imp = torch.zeros(imp.shape)[~t_labeled_set]
        for t in range(self.time_span):
            #未标记的id
            ids = np.arange(self.node_num)[~t_labeled_set[t*self.node_num:(t+1)*self.node_num]]
            #标记了的id
            label_ids = np.arange(self.node_num)[t_labeled_set[t*self.node_num:(t+1)*self.node_num]]
            #标记节点与所有的相关节点
            all_l_neighbors = []
            for lid in label_ids:
                lid_ = t*self.node_num+lid
                l_neighbors = [neighbor+t*self.node_num for neighbor in neighbor_dics[t][lid]]
                if (lid_ in l_neighbors) == False:
                    l_neighbors.append(lid_)
                all_l_neighbors.extend(l_neighbors)
            all_l_neighbors = list(set(all_l_neighbors))
            #print(ids)
            for id in ids:
                #获取neighbor在全时间段的ID
                id_ = t*self.node_num+id
                neighbors = [neighbor+t*self.node_num for neighbor in neighbor_dics[t][id] if t_labeled_set[t*self.node_num+neighbor] == False]
                if (id_ in neighbors) == False:
                    neighbors.append(id_)
                #获取节点id在非标记节点集上的位置
                q = len(np.arange(id_)[~t_labeled_set[0:id_]].tolist())
                #add_imp[q] = sum([imp[i] for i in neighbors if (((i in all_l_neighbors)==False) and (i==id_ or imp[i]>=0))])
                add_imp[q] = sum(
                    [imp[i] for i in neighbors if i not in all_l_neighbors])
                #print('333')
                #print(imp[neighbors])
                #print(add_imp[q])
                #exit()
        return add_imp
    def neighbor_add_imp_in_time(self, t_labeled_set, neighbor_dics, add_imp, imp, t):
        #neighbor_dic 根据时间，节点获取邻居
        #print(neighbor_dics)
        add_imp = add_imp
        removed_imp_ids = np.where(add_imp == -100)[0]
        #print(removed_imp_ids)
        #add_imp = torch.zeros(add_imp.shape)[~t_labeled_set]
        if True:
            #未标记的id
            ids = np.arange(self.node_num)[~t_labeled_set[t*self.node_num:(t+1)*self.node_num]]
            #标记了的id
            label_ids = np.arange(self.node_num)[t_labeled_set[t*self.node_num:(t+1)*self.node_num]]
            #标记节点与所有的相关节点
            all_l_neighbors = []
            for lid in label_ids:
                lid_ = t*self.node_num+lid
                l_neighbors = [neighbor+t*self.node_num for neighbor in neighbor_dics[t][lid]]
                if (lid_ in l_neighbors) == False:
                    l_neighbors.append(lid_)
                all_l_neighbors.extend(l_neighbors)
            all_l_neighbors = list(set(all_l_neighbors))
            #print(ids)
            for id in ids:
                #获取neighbor在全时间段的ID
                id_ = t*self.node_num+id
                neighbors = [neighbor+t*self.node_num for neighbor in neighbor_dics[t][id] if t_labeled_set[t*self.node_num+neighbor] == False]
                if (id_ in neighbors) == False:
                    neighbors.append(id_)
                #获取节点id在非标记节点集上的位置
                q = len(np.arange(id_)[~t_labeled_set[0:id_]].tolist())
                #add_imp[q] = sum([imp[i] for i in neighbors if (((i in all_l_neighbors)==False) and (i==id_ or imp[i]>=0))])
                add_imp[q] = sum(
                    [imp[i] for i in neighbors if i not in all_l_neighbors])
        add_imp[removed_imp_ids] =-100
                #print('333')
                #print(imp[neighbors])
                #print(add_imp[q])
                #exit()
        return add_imp
    #q_idx为增加的节点
    def change_imp_after_delete(self, add_imp, neighbor_dics, q_idx):
        neighbors = []
        t = q_idx//self.node_num
        neighbors = neighbor_dics[t][q_idx%self.node_num]+t*self.node_num







