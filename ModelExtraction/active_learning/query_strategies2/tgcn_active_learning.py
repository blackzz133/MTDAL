#mixed and time-sensitive temporal active learning
import numpy as np
import networkx as nx
import torch
import math
from collections import Counter
from tqdm import tqdm
from sklearn.cluster import KMeans
from ModelExtraction.active_learning.utils import SupervisedKMeans, SemiKMeans
class MTTAL_strategy:
    def __init__(self, dataset, models, labeled_set, cycle, all_loss, args, query_cost, victim_labels, selected_ids = None):
        self.dataset = dataset
        self.models = models
        self.labeled_set = labeled_set
        self.cycle = cycle
        self.budget = args.budget
        self.all_loss = all_loss
        self.time_span = torch.tensor(labeled_set).shape[0]
        self.node_num = torch.tensor(labeled_set).shape[1]
        self.total_cycle = round(self.time_span * args.budget / args.queries)
        #self.Rep = Rep #节点的representative
        #self.cluster_ids_x = cluster_ids_x
        self.victim_labels = victim_labels
        self.class_num = args.class_num
        self.query_cost = query_cost
        self.selected_ids = selected_ids
        #query_cost is (t,n,f)
    #without query cost
    def query(self, n):
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
        a = self.Rep
        b = self.inf()  ###################################有问题
        # print('wwwwww')
        # print(a)
        # print(b)
        imp = alph * a + (1 - alph) * b

        unlabeled_imp = imp[~t_labeled_set]
        edge_indices = self.dataset.edge_indices
        neighbor_dics = []
        for i in range(self.time_span):
            graph = nx.Graph()
            graph.add_nodes_from([j for j in range(self.node_num)])
            edges = torch.tensor(edge_indices[i]).t().tolist()
            graph.add_edges_from(edges)
            neighbor_dic = [[k for k in graph.neighbors(j)] for j in range(self.node_num)]
            neighbor_dics.append(neighbor_dic)
        unlabeled_neighbor_imp = self.temporal_neighbor_add_imp(t_labeled_set, neighbor_dics, imp)

        # exit()
        count = Counter(self.cluster_ids_x[t_labeled_set].tolist())

        # constraint of selected node
        c = (len(np.arange(self.time_span * self.node_num)[t_labeled_set].tolist()) + n) / self.class_num
        for i in tqdm(range(n), ncols=100):
            if used_budget + qcost_min > n:
                break
            # 返回各行的和
            l = -1
            # removed_ids = []
            # removed_imps = []
            np.argmax()
            while l == -1:
                q_idx_ = (unlabeled_neighbor_imp / qcost).argmax()
                if unlabeled_neighbor_imp[q_idx_] == -100:
                    break
                if used_budget + qcost[q_idx_] > n:
                    # imp[q_idx] = -100
                    # unlabeled_imp[q_idx_] = -100
                    unlabeled_neighbor_imp[q_idx_] = -100
                    continue
                # 获取最大的节点所在位置
                q_idx = np.arange(self.labeled_set.shape[0] * self.labeled_set.shape[1])[~t_labeled_set][q_idx_]
                # print(q_idx)
                class_q = self.cluster_ids_x[q_idx]
                # 某类数据选择后,该类数据的数量如果大于c
                if count[class_q] + 1 > c:
                    # 暂时存储不选的
                    unlabeled_neighbor_imp[q_idx_] = -100
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
                    unlabeled_neighbor_imp[_start:_end] = torch.ones(_end - _start) * -100
                    # unlabeled_neighbor_imp[q_idx_] = -100
                    # unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    continue
                else:
                    t_labeled_set[q_idx] = True
                    l = 0
                    count[class_q] += 1
                    used_budget += t_query_cost[q_idx_]
                    # unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
                    # 每次选择多个
                    unlabeled_neighbor_imp = np.delete(unlabeled_neighbor_imp, q_idx_, 0)
                    qcost = t_query_cost[~t_labeled_set]
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

    #with the query_cost
    def query2(self, n):
        print('query of this cycle is started')
        used_budget =0
        lset = torch.tensor(self.labeled_set)
        t_query_cost = self.query_cost.reshape(lset.shape[0] * lset.shape[1]).numpy()
        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()
        qcost = t_query_cost[~t_labeled_set]
        qcost_min = np.array(qcost).min()
        # the weight of representative
        alph = math.cos(math.pi*(self.cycle+1)/(2*self.total_cycle))

        # the used cost in each snapshot
        cost_counts = [float(sum(self.query_cost[i][self.labeled_set[i]])) for i in range(self.time_span)]
        #label_counts = [len(np.ones(lset.shape[1])[lset[i]].tolist()) for i in range(lset.shape[0])]
        #std = np.std(label_counts, axis=0)
        std = np.std(cost_counts, axis=0)
        #max_count = self.budget
        # Range Rule for Standard Deviation
        #est_std = (max_count - n)/4
        est_std = ((max(cost_counts)- min(cost_counts))/4)
        # e^-ax/(x+1) a is bigger, the value has more variation, a= p(std/(est_std))
        #beta1 1-预算消耗率。
        #print(sum(t_query_cost[t_labeled_set])/(self.time_span*self.budget*2))
        #print(self.time_span*self.budget)
        # p2 controls the beta1,must greater than 1
        p1 = 2
        #beta1 = min(1, sum(t_query_cost[t_labeled_set])/(self.time_span*self.budget)*p1)
        beta1 = math.cos(math.pi * sum(t_query_cost[t_labeled_set]) / (2 * self.budget * self.time_span))
        #print('beta1')
        #print(beta1)
        #print('std','est_std')
        #print(std, est_std)
        p2 = 0.1
        #beta2: std与值大小
        if est_std == 0:
            beta2 = torch.zeros(self.time_span)
            #beta2 = torch.ones(self.time_span) #ones
        else:
            #beta2 = torch.zeros(self.time_span)
            beta2 = torch.zeros(self.time_span)
            #beta2 = torch.tensor([math.exp(-(p2 * std / est_std)) / (cost_counts[i] - min(cost_counts) + 1) for i in range(self.time_span)])
        #beta2 = torch.tensor([math.exp(-(p*std/est_std)*(cost_counts[i] - min(cost_counts)))/(cost_counts[i]-min(cost_counts)+1) for i in range(self.time_span)])

        # imp: importance of nodes- include rep+inf
        # rep: get representative
        #print('beta2')
        #print(beta2)
        #a = self.rep()
        #is Rep
        cluster_ids_x, Rep = self.rep(self.models['FREE'], self.dataset)
        cluster_ids_x2, gcn_rep = self.rep2(self.models['GCN'], self.dataset)
        gcn_rep = gcn_rep.reshape(self.time_span,self.node_num)
        # select type of RGCN based on all loss
        res = max(self.all_loss, key=lambda x: self.all_loss[x])
        #print(res)
        cluster_ids_x3, rgcn_rep = self.rep2(self.models[res], self.dataset)
        rgcn_rep = rgcn_rep.reshape(self.time_span,self.node_num)
        #print('beta1,beta2')
        #print(beta1)
        #print(beta2)
        #print(gcn_rep)
        #print(rgcn_rep)
        '''
        
        print(self.Rep.shape)
        print(gcn_rep.shape)
        print(rgcn_rep.shape)
        print('######')
        print((beta1*self.Rep).shape)
        print(gcn_rep.shape)
        print(beta2.shape)
        print((beta2*gcn_rep).shape)
        print(((1-beta2)*rgcn_rep).shape)
        exit()
        '''
        a1 = beta1*torch.tensor([(beta2[i]* gcn_rep[i]).tolist() for i in range(self.time_span)]).reshape(self.time_span*self.node_num)
        a2 = beta1*torch.tensor([((1-beta2[i])* rgcn_rep[i]).tolist() for i in range(self.time_span)]).reshape(self.time_span*self.node_num)
        #print('a1,a2')
        #print(a1)  #tensor([0.4841, 0.4841, 0.6326,  ..., 0.7069, 0.4410, 0.5203])
        #print(a2) #tensor([0., 0., 0.,  ..., 0., 0., 0.])
        a = (1-beta1)*Rep + a1 +a2
        s = sorted(np.unique(a))
        s_ = {float(s[i]): i for i in range(len(s))}
        new_a = [s_[float(a[i])] for i in range(len(a))]
        new_a_ = np.array(new_a) / max(new_a)
        #print(new_a) #[883, 883, 5201, 4622, 5201, 4886, 3047, 3865, 5669, 1857, 1857, 2151, 5812, 7681, 3799, 2850, 3692, 530, 3692, 1649, 162, 162, 809...
        #print(max(new_a)) #8091
        #print(new_a_) #[0.10913361 0.10913361 0.642813   ... 0.99987641 0.04066246 0.042022  ]
        #inf: get informative
        #print('b')
        #测试中，先不考虑inf,只考虑rep
        b = self.temp_inf()
        s2 = sorted(np.unique(b))
        #print(s2)
        s2_ = {float(s2[i]): i for i in range(len(s2))}
        new_b = [s2_[float(b[i])] for i in range(len(b))]
        new_b_ = np.array(new_b) / max(new_b)


        #imp = alph/(max(new_a)-new_a_+1)+(1-alph)/(max(new_b)-new_b_+1)
        #imp = alph*a +(1-alph)*b
        #imp = a+b
        imp = new_a_+new_b_
        '''
        print(torch.mean(imp[0:10000]))
        print(max(imp[0:10000]))
        print(max(imp[10000:20000]))
        print(max(imp[20000:30000]))
        print(a[0:10000])
        print(alph*a[0:10000])
        print(b[0:10000])
        print((1-alph)*b[0:10000])
        print(imp[0:10000])

        print(torch.mean(imp[0:10000]))

        '''
        unlabeled_imp = imp[~t_labeled_set]
        edge_indices = self.dataset.edge_indices
        neighbor_dics = []
        for i in range(self.time_span):
            graph = nx.Graph()
            graph.add_nodes_from([j for j in range(self.node_num)])
            edges = torch.tensor(edge_indices[i]).t().tolist()
            graph.add_edges_from(edges)
            neighbor_dic = [[k for k in graph.neighbors(j)] for j in range(self.node_num)]
            neighbor_dics.append(neighbor_dic)
        unlabeled_neighbor_imp = self.temporal_neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
        # exit()
        count = {i: 0 for i in range(self.class_num)}
        for i in range(cluster_ids_x[t_labeled_set].shape[0]):
            count[cluster_ids_x[i]] += t_query_cost[t_labeled_set][i]
        # constraint of selected node
        c = n*(self.cycle+1)/self.class_num
        #budget_end_list = np.zeros(self.time_span, dtype=bool)
        #label for determining whether is iteration end
        #p = -1
        for i in tqdm(range(n), ncols=100):
            #if len(np.ones(self.time_span)[budget_end_list]) ==0:
                #break
            if used_budget + qcost_min > n :#or p == 0
                break
            # 返回各行的和
            # l 判断循环的结束， p判断为0的时候, q判断
            l = -1
            while l == -1:
                q_idx_ = (unlabeled_neighbor_imp / qcost).argmax()
                if unlabeled_neighbor_imp[q_idx_] == -100 or unlabeled_neighbor_imp[q_idx_] == -200:
                    break
                if used_budget + qcost[q_idx_] > n:
                    unlabeled_neighbor_imp[q_idx_] = -200
                    continue
                # 获取最大的节点所在位置
                q_idx = np.arange(self.labeled_set.shape[0] * self.labeled_set.shape[1])[~t_labeled_set][q_idx_]
                # cluster_ids需要改成模型推导的label
                class_q = cluster_ids_x3[q_idx]
                # 某类数据选择后,该类数据的数量如果大于c
                if count[class_q]+qcost[q_idx_] >c:
                    # 暂时存储不选的
                    unlabeled_neighbor_imp[q_idx_] = -200
                    continue
                start = q_idx // self.node_num * self.node_num
                end = start + self.node_num
                _start = start - np.arange(start)[t_labeled_set[0:start]].shape[0]
                _end = end - np.arange(end)[t_labeled_set[0:end]].shape[0]
                #if len(np.ones(self.node_num)[t_labeled_set[start:end]].tolist()) >= self.budget \
                if sum(t_query_cost[start:end][t_labeled_set[start:end]]) + min(
                        t_query_cost[start:end][~t_labeled_set[start:end]]) > self.budget:
                    # unlabeled_imp[q_idx_] = 0
                    # 区间内所有imp值归0，以后不再选择
                    # imp[start:end] = torch.tensor([-100 for i in range(start, end)])
                    # unlabeled_imp = imp[~t_labeled_set]
                    unlabeled_neighbor_imp[_start:_end] = torch.ones(_end - _start) * -100
                    # unlabeled_neighbor_imp[q_idx_] = -100
                    # unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    continue
                elif sum(t_query_cost[start:end][t_labeled_set[start:end]])+t_query_cost[q_idx]>self.budget:
                    unlabeled_neighbor_imp[q_idx_] = -100
                    continue
                else:
                    t_labeled_set[q_idx] = True
                    l = 0
                    count[class_q] += qcost[q_idx_]
                    used_budget += t_query_cost[q_idx]
                    # unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
                    # 每次选择多个
                    unlabeled_neighbor_imp = np.delete(unlabeled_neighbor_imp, q_idx_, 0)
                    qcost = t_query_cost[~t_labeled_set]
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

    #with selected_ids
    def query3(self, n):
        print('query of this cycle is started')
        selected_ids = list(set(self.selected_ids))

        used_budget = 0
        lset = torch.tensor(self.labeled_set)
        t_query_cost = self.query_cost.reshape(lset.shape[0] * lset.shape[1]).numpy()


        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()
        qcost = t_query_cost[~t_labeled_set]
        qcost_min = np.array(qcost).min()
        selected_q_cost = t_query_cost[selected_ids]
        # the weight of representative
        alph = math.cos(math.pi * (self.cycle + 1) / (2 * self.total_cycle))

        # the used cost in each snapshot
        cost_counts = [float(sum(self.query_cost[i][self.labeled_set[i]])) for i in range(self.time_span)]
        # label_counts = [len(np.ones(lset.shape[1])[lset[i]].tolist()) for i in range(lset.shape[0])]
        # std = np.std(label_counts, axis=0)
        std = np.std(cost_counts, axis=0)
        # max_count = self.budget
        # Range Rule for Standard Deviation
        # est_std = (max_count - n)/4
        est_std = ((max(cost_counts) - min(cost_counts)) / 4)
        # e^-ax/(x+1) a is bigger, the value has more variation, a= p(std/(est_std))
        # beta1 1-预算消耗率。
        # print(sum(t_query_cost[t_labeled_set])/(self.time_span*self.budget*2))
        # print(self.time_span*self.budget)
        # p2 controls the beta1,must greater than 1
        p1 = 2
        # beta1 = min(1, sum(t_query_cost[t_labeled_set])/(self.time_span*self.budget)*p1)
        beta1 = math.cos(math.pi * sum(t_query_cost[t_labeled_set]) / (2 * self.budget * self.time_span))
        # print('beta1')
        # print(beta1)
        # print('std','est_std')
        # print(std, est_std)
        p2 = 0.1
        # beta2: std与值大小
        if est_std == 0:
            beta2 = torch.zeros(self.time_span)
            # beta2 = torch.ones(self.time_span) #ones
        else:
            # beta2 = torch.zeros(self.time_span)
            beta2 = torch.zeros(self.time_span)
            # beta2 = torch.tensor([math.exp(-(p2 * std / est_std)) / (cost_counts[i] - min(cost_counts) + 1) for i in range(self.time_span)])
        # beta2 = torch.tensor([math.exp(-(p*std/est_std)*(cost_counts[i] - min(cost_counts)))/(cost_counts[i]-min(cost_counts)+1) for i in range(self.time_span)])

        # imp: importance of nodes- include rep+inf
        # rep: get representative
        # print('beta2')
        # print(beta2)
        # a = self.rep()
        # is Rep
        cluster_ids_x, Rep = self.rep(self.models['FREE'], self.dataset)
        cluster_ids_x2, gcn_rep = self.rep2(self.models['GCN'], self.dataset)
        gcn_rep = gcn_rep.reshape(self.time_span, self.node_num)
        # select type of RGCN based on all loss
        res = max(self.all_loss, key=lambda x: self.all_loss[x])
        # print(res)
        cluster_ids_x3, rgcn_rep = self.rep2(self.models[res], self.dataset)
        rgcn_rep = rgcn_rep.reshape(self.time_span, self.node_num)
        # print('beta1,beta2')
        # print(beta1)
        # print(beta2)
        # print(gcn_rep)
        # print(rgcn_rep)
        '''

        print(self.Rep.shape)
        print(gcn_rep.shape)
        print(rgcn_rep.shape)
        print('######')
        print((beta1*self.Rep).shape)
        print(gcn_rep.shape)
        print(beta2.shape)
        print((beta2*gcn_rep).shape)
        print(((1-beta2)*rgcn_rep).shape)
        exit()
        '''
        a1 = beta1 * torch.tensor([(beta2[i] * gcn_rep[i]).tolist() for i in range(self.time_span)]).reshape(
            self.time_span * self.node_num)
        a2 = beta1 * torch.tensor([((1 - beta2[i]) * rgcn_rep[i]).tolist() for i in range(self.time_span)]).reshape(
            self.time_span * self.node_num)
        # print('a1,a2')
        # print(a1)  #tensor([0.4841, 0.4841, 0.6326,  ..., 0.7069, 0.4410, 0.5203])
        # print(a2) #tensor([0., 0., 0.,  ..., 0., 0., 0.])
        a = (1 - beta1) * Rep + a1 + a2
        s = sorted(np.unique(a))
        s_ = {float(s[i]): i for i in range(len(s))}
        new_a = [s_[float(a[i])] for i in range(len(a))]
        new_a_ = np.array(new_a) / max(new_a)
        # print(new_a) #[883, 883, 5201, 4622, 5201, 4886, 3047, 3865, 5669, 1857, 1857, 2151, 5812, 7681, 3799, 2850, 3692, 530, 3692, 1649, 162, 162, 809...
        # print(max(new_a)) #8091
        # print(new_a_) #[0.10913361 0.10913361 0.642813   ... 0.99987641 0.04066246 0.042022  ]
        # inf: get informative
        # print('b')
        # 测试中，先不考虑inf,只考虑rep
        b = self.temp_inf()
        s2 = sorted(np.unique(b))
        # print(s2)
        s2_ = {float(s2[i]): i for i in range(len(s2))}
        new_b = [s2_[float(b[i])] for i in range(len(b))]
        new_b_ = np.array(new_b) / max(new_b)

        # imp = alph/(max(new_a)-new_a_+1)+(1-alph)/(max(new_b)-new_b_+1)
        # imp = alph*a +(1-alph)*b
        # imp = a+b
        imp = new_a_ + new_b_
        '''
        print(torch.mean(imp[0:10000]))
        print(max(imp[0:10000]))
        print(max(imp[10000:20000]))
        print(max(imp[20000:30000]))
        print(a[0:10000])
        print(alph*a[0:10000])
        print(b[0:10000])
        print((1-alph)*b[0:10000])
        print(imp[0:10000])

        print(torch.mean(imp[0:10000]))

        '''
        #unlabeled_imp = imp[~t_labeled_set]
        #selected_imp = imp[selected_ids]
        # output the selected_indexes_in_unlabeled
        ww = np.arange(self.node_num*self.time_span)[~t_labeled_set]
        selected_index_in_unlabeled = [int(np.where(ww==id)[0]) for id in selected_ids]
        unselected_index_in_unlabeled = np.delete(np.arange(ww.shape[0]), selected_index_in_unlabeled).tolist()
        edge_indices = self.dataset.edge_indices
        neighbor_dics = []
        for i in range(self.time_span):
            graph = nx.Graph()
            graph.add_nodes_from([j for j in range(self.node_num)])
            edges = torch.tensor(edge_indices[i]).t().tolist()
            graph.add_edges_from(edges)
            neighbor_dic = [[k for k in graph.neighbors(j)] for j in range(self.node_num)]
            neighbor_dics.append(neighbor_dic)
        unlabeled_neighbor_imp = self.temporal_neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
        #print(unlabeled_neighbor_imp[unselected_index_in_unlabeled].shape)
        #print((torch.ones(len(unselected_index_in_unlabeled))*-100).shape)
        for id in unselected_index_in_unlabeled:
            unlabeled_neighbor_imp[id] = -100

        # exit()
        count = {i: 0 for i in range(self.class_num)}
        for i in range(cluster_ids_x[t_labeled_set].shape[0]):
            count[cluster_ids_x[i]] += t_query_cost[t_labeled_set][i]
        # constraint of selected node
        c = n * (self.cycle + 1) / self.class_num
        # budget_end_list = np.zeros(self.time_span, dtype=bool)
        # label for determining whether is iteration end
        # p = -1
        for i in tqdm(range(n), ncols=100):
            # if len(np.ones(self.time_span)[budget_end_list]) ==0:
            # break
            if used_budget + qcost_min > n:  # or p == 0
                break
            # 返回各行的和
            # l 判断循环的结束， p判断为0的时候, q判断
            l = -1
            while l == -1:
                q_idx_ = (unlabeled_neighbor_imp / qcost).argmax()
                if unlabeled_neighbor_imp[q_idx_] == -100 or unlabeled_neighbor_imp[q_idx_] == -200:
                    break
                if used_budget + qcost[q_idx_] > n:
                    unlabeled_neighbor_imp[q_idx_] = -200
                    continue
                # 获取最大的节点所在位置
                q_idx = np.arange(self.labeled_set.shape[0] * self.labeled_set.shape[1])[~t_labeled_set][q_idx_]
                # cluster_ids需要改成模型推导的label
                class_q = cluster_ids_x3[q_idx]
                # 某类数据选择后,该类数据的数量如果大于c
                if count[class_q] + qcost[q_idx_] > c:
                    # 暂时存储不选的
                    unlabeled_neighbor_imp[q_idx_] = -200
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
                    unlabeled_neighbor_imp[_start:_end] = torch.ones(_end - _start) * -100
                    # unlabeled_neighbor_imp[q_idx_] = -100
                    # unlabeled_neighbor_imp = self.neighbor_add_imp(t_labeled_set, neighbor_dics, imp)
                    continue
                elif sum(t_query_cost[start:end][t_labeled_set[start:end]]) + t_query_cost[q_idx] > self.budget:
                    unlabeled_neighbor_imp[q_idx_] = -100
                    continue
                else:
                    t_labeled_set[q_idx] = True
                    l = 0
                    count[class_q] += qcost[q_idx_]
                    used_budget += t_query_cost[q_idx]
                    # unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
                    # 每次选择多个
                    unlabeled_neighbor_imp = np.delete(unlabeled_neighbor_imp, q_idx_, 0)
                    qcost = t_query_cost[~t_labeled_set]
                    unlabeled_neighbor_imp = self.neighbor_add_imp_in_time(t_labeled_set, neighbor_dics,
                                                                           unlabeled_neighbor_imp, imp,
                                                                           q_idx // self.node_num)
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
    # 一维和二维已经融合
    def inf(self):
        all_type_embeddings = {}
        INF = []
        for key, value in self.models.items():
            all_embeddings = []
            if key!='FREE' and key!='victim':
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
        for i in range(self.labeled_set.shape[0]*self.labeled_set.shape[1]):
            result = 0
            for key, value in self.all_loss.items():
                for key2, value2 in self.all_loss.items():
                    #print(key, value)
                    #print(key2, value2)
                    error1 = 0.5 * torch.log2((1-value)/value)
                    error2 = 0.5 * torch.log2((1-value2)/value2)
                    result += error1*error2*torch.dist(all_type_embeddings[key][i], all_type_embeddings[key2][i])
            INF.append(result)
        return torch.tensor(INF)

    def temp_inf(self):
        all_type_embeddings = {}
        INF = []
        for key, value in self.models.items():
            all_embeddings = []
            if key != 'FREE' and key != 'victim':
                for time, snapshot in enumerate(self.dataset):
                    model = self.models[key].cuda()
                    x = snapshot.x.cuda()
                    edge_index = snapshot.edge_index.cuda()
                    edge_attr = snapshot.edge_attr.cuda()
                    embeddings = model(x, edge_index, edge_attr).detach()
                    # print(embeddings)
                    embeddings = embeddings.tolist()
                    all_embeddings.append(embeddings)
                all_embeddings = torch.tensor(all_embeddings)
                all_embeddings = all_embeddings.reshape(self.labeled_set.shape[0] * self.labeled_set.shape[1],
                                                        all_embeddings.shape[2])
                all_type_embeddings[key] = all_embeddings
        for i in range(self.labeled_set.shape[0] * self.labeled_set.shape[1]):
            result = 0
            result2 = 0
            for key, value in self.all_loss.items():
                for key2, value2 in self.all_loss.items():
                    # print(key, value)
                    # print(key2, value2)
                    error1 = 0.5 * torch.log2((1 - value) / value)
                    error2 = 0.5 * torch.log2((1 - value2) / value2)
                    result += error1 * error2 * torch.dist(all_type_embeddings[key][i], all_type_embeddings[key2][i])
                    if i < self.node_num:
                        result2 += abs(
                            torch.dist(all_type_embeddings[key][i], all_type_embeddings[key2][i + self.node_num]-
                                       torch.dist(all_type_embeddings[key2][i], all_type_embeddings[key][i + self.node_num])))
                    elif i+self.node_num > self.node_num*self.time_span-1:
                        result2 += abs(
                            torch.dist(all_type_embeddings[key][i], all_type_embeddings[key2][i - self.node_num]-
                                       torch.dist(all_type_embeddings[key2][i], all_type_embeddings[key][i - self.node_num])))
                    else:
                        result2 += abs(torch.dist(all_type_embeddings[key2][i], all_type_embeddings[key][i+self.node_num]-
                                                                    torch.dist(all_type_embeddings[key][i], all_type_embeddings[key2][i+self.node_num])))
                        result2 += abs(torch.dist(all_type_embeddings[key2][i], all_type_embeddings[key][i-self.node_num]-
                                                                    torch.dist(all_type_embeddings[key][i], all_type_embeddings[key2][i-self.node_num])))
                        result2 = error1 * error2* result2/2

            INF.append(result+result2)
            #INF.append(result)
        #new temporal inf
        return torch.tensor(INF)

    def rep(self, model, dataset):
        lset = torch.tensor(self.labeled_set)
        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()
        #dataset = dataloaders['sim_train']
        #raw_dataset = dataloaders['raw_train']
        # 转换list
        model_ = model.to('cpu')
        num_clusters = self.class_num
        all_embeddings = []
        for time, snapshot in enumerate(dataset):
            x = snapshot.x
            edge_index = snapshot.edge_index
            edge_attr = snapshot.edge_attr
            embeddings = model_(x.to('cpu'), edge_index.to('cpu'), edge_attr.to('cpu')).detach()
            embeddings = embeddings.tolist()
            all_embeddings.append(embeddings)
        all_embeddings = torch.tensor(all_embeddings).to('cpu')
        all_embeddings = all_embeddings.reshape(int(all_embeddings.shape[0] * all_embeddings.shape[1]),
                                                all_embeddings.shape[2])
        y0 = np.array(self.victim_labels).reshape(self.time_span * self.node_num)[t_labeled_set]
        x0 = all_embeddings[t_labeled_set].numpy()
        t0 = np.arange(self.time_span * self.node_num)[t_labeled_set].tolist()
        t1 = np.arange(self.time_span * self.node_num)[~t_labeled_set].tolist()
        x1 = all_embeddings[~t_labeled_set].numpy()
        km = SemiKMeans(n_clusters=num_clusters)
        km.fit(x0, y0, x1)
        y1 = km.predict(x1)
        #print('3333')
        #print(y1)
        cluster_ids_x = np.ones(self.time_span * self.node_num, dtype=int)
        cluster_ids_x[t0] = y0
        cluster_ids_x[t1] = y1
        cluster_ids_x = np.array([int(i) for i in cluster_ids_x])
        cluster_centers = torch.tensor(km.cluster_centers_)
        Rep = 1 - torch.abs(torch.tensor(
            [torch.dist(all_embeddings[i], cluster_centers[cluster_ids_x[i]]) for i in range(all_embeddings.shape[0])]))
        #print(Rep)

        max_rep = [-100 for i in range(self.class_num)]
        min_rep = [100 for i in range(self.class_num)]
        for i in range(Rep.shape[0]):
            max_rep[cluster_ids_x[i]] = max(max_rep[cluster_ids_x[i]], Rep[i])
            min_rep[cluster_ids_x[i]] = min(min_rep[cluster_ids_x[i]], Rep[i])
        Rep = torch.tensor(
            [(Rep[i] - min_rep[cluster_ids_x[i]]) / (max_rep[cluster_ids_x[i]] - min_rep[cluster_ids_x[i]]) for i in
             range(Rep.shape[0])])
        #print(Rep)
        return cluster_ids_x, Rep

    def rep2(self, model, dataset):
        lset = torch.tensor(self.labeled_set)
        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()
        #dataset = dataloaders['sim_train']
        #raw_dataset = dataloaders['raw_train']
        # 转换list
        model_ = model.cuda()
        num_clusters = self.class_num
        all_embeddings = []
        for time, snapshot in enumerate(dataset):
            x = snapshot.x
            edge_index = snapshot.edge_index
            edge_attr = snapshot.edge_attr
            embeddings = model_(x.cuda(), edge_index.cuda(), edge_attr.cuda()).detach()
            embeddings = embeddings.tolist()
            all_embeddings.append(embeddings)
        all_embeddings = torch.tensor(all_embeddings).to('cpu')
        all_embeddings = all_embeddings.reshape(int(all_embeddings.shape[0] * all_embeddings.shape[1]),
                                                all_embeddings.shape[2])
        y0 = np.array(self.victim_labels).reshape(self.time_span * self.node_num)[t_labeled_set]
        x0 = all_embeddings[t_labeled_set].numpy()
        t0 = np.arange(self.time_span * self.node_num)[t_labeled_set].tolist()
        t1 = np.arange(self.time_span * self.node_num)[~t_labeled_set].tolist()
        x1 = all_embeddings[~t_labeled_set].numpy()
        km = SemiKMeans(n_clusters=num_clusters)
        km.fit(x0, y0, x1)
        y1 = km.predict(x1)
        cluster_ids_x = np.ones(self.time_span * self.node_num, dtype=int)
        cluster_ids_x[t0] = y0
        cluster_ids_x[t1] = y1
        cluster_ids_x = np.array([int(i) for i in cluster_ids_x])
        cluster_centers = torch.tensor(km.cluster_centers_)
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



    def temporal_neighbor_add_imp(self, t_labeled_set, neighbor_dics, imp):
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
                if t == 0:
                    past_neighbors = []
                    next_neighbors = [neighbor + (t + 1) * self.node_num for neighbor in neighbor_dics[t + 1][id] if
                                      t_labeled_set[(t + 1) * self.node_num + neighbor] == False]
                elif t == self.time_span - 1:
                    past_neighbors = [neighbor + (t - 1) * self.node_num for neighbor in neighbor_dics[t - 1][id] if
                                      t_labeled_set[(t - 1) * self.node_num + neighbor] == False]
                    next_neighbors = []
                else:
                    past_neighbors = [neighbor + (t - 1) * self.node_num for neighbor in neighbor_dics[t - 1][id] if
                                      t_labeled_set[(t - 1) * self.node_num + neighbor] == False]
                    next_neighbors = [neighbor + (t + 1) * self.node_num for neighbor in neighbor_dics[t + 1][id] if
                                      t_labeled_set[(t + 1) * self.node_num + neighbor] == False]
                if (id_ in neighbors) == False:
                    neighbors.append(id_)
                if (id_ in past_neighbors) == False:
                    neighbors.append(id_)
                if (id_ in next_neighbors) == False:
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
                if t == 0:
                    past_neighbors = []
                    next_neighbors = [neighbor + (t + 1) * self.node_num for neighbor in neighbor_dics[t + 1][id] if
                                      t_labeled_set[(t + 1) * self.node_num + neighbor] == False]
                elif t == self.time_span - 1:
                    past_neighbors = [neighbor + (t - 1) * self.node_num for neighbor in neighbor_dics[t - 1][id] if
                                      t_labeled_set[(t - 1) * self.node_num + neighbor] == False]
                    next_neighbors = []
                else:
                    past_neighbors = [neighbor + (t - 1) * self.node_num for neighbor in neighbor_dics[t - 1][id] if
                                      t_labeled_set[(t - 1) * self.node_num + neighbor] == False]
                    next_neighbors = [neighbor + (t + 1) * self.node_num for neighbor in neighbor_dics[t + 1][id] if
                                      t_labeled_set[(t + 1) * self.node_num + neighbor] == False]
                if (id_ in neighbors) == False:
                    neighbors.append(id_)
                if (id_ in past_neighbors) == False:
                    neighbors.append(id_)
                if (id_ in next_neighbors) == False:
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
