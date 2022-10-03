import numpy as np
import torch

from .strategy import Strategy
from .strategy2 import Strategy2
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import math
'''
class KCenterGreedy(Strategy):
    def __init__(self, dataset, net):
        super(KCenterGreedy, self).__init__(dataset, net)

    def query(self, n):
        labeled_idxs, train_data = self.dataset.get_train_data()
        embeddings = self.get_embeddings(train_data)
        embeddings = embeddings.numpy()

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
            
        return np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]

'''
class KCenterGreedy:
    def __init__(self, dataset, models, labeled_set, budget, query_cost):
        self.dataset = dataset
        self.models = models
        self.labeled_set = labeled_set
        self.budget = budget
        self.query_cost = query_cost
    #without query_cost
    def query(self, n, query_type):
        lset = torch.tensor(self.labeled_set)
        time_span = torch.tensor(self.dataset.targets).shape[0]
        node_num = torch.tensor(self.dataset.targets).shape[1]
        qcost = []
        mat_all = []
        dist_mat_all = []
        model = self.models[query_type].to('cpu')
        for time, snapshot in enumerate(self.dataset):
            x = snapshot.x
            n_pool = x.shape[0]
            edge_index = snapshot.edge_index
            edge_weight = snapshot.edge_attr
            labeled_idxs = self.labeled_set[time]
            embeddings = self.models[query_type](x, edge_index, edge_weight)
            embeddings = embeddings.detach().numpy()
            dist_mat = np.matmul(embeddings, embeddings.transpose())
            sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
            dist_mat *= -2
            dist_mat += sq
            dist_mat += sq.transpose()
            dist_mat = np.sqrt(np.abs(dist_mat))
            # ~labeled_idxs 选择标FALSE的行表示未标注的节点，labeled_idxs选择标True的列表示已标注的节点
            # mat_all用来装非标记节点 到节点的距离，dis_mat_all放所有节点到所有节点的距离
            mat_all.append(dist_mat[~labeled_idxs, :][:, labeled_idxs])
            qcost.append(self.query_cost[time].numpy()[~labeled_idxs])
            dist_mat_all.append(dist_mat)
        #mat_all = np.array(mat_all)
        #dist_mat_all = np.array(dist_mat_all)
        #找出需要标记的点位置与时间
        for q in tqdm(range(n), ncols=100):
            t = 0
            # 返回各行的和
            l = -1
            while l == -1:
                mat_min = [np.array(i).min(axis=1) for i in mat_all]
                #print('111')
                t = np.array([np.max(i) for i in mat_min]).argmax()
                #局部最大值
                q_idx_= np.argmax(mat_min[t])
                #全局最大值
                q_idx = np.arange(node_num)[~lset[t]][q_idx_]
                #print('222')
                # 如果超过了预算
                if len(np.ones(node_num)[lset[t]].tolist()) >= self.budget:
                    # mat_min[q_idx_] = 0
                    dist_mat_all[t] = np.zeros(dist_mat_all[t].shape)
                    mat_all[t] = np.zeros(mat_all[t].shape)
                    # mat_all = dist_mat_all[~t_labeled_set]
                    continue
                else:
                    # 没超过越算
                    lset[t][q_idx] = True
                    l = 0
            mat_all[t] = np.delete(mat_all[t], q_idx_, 0)
            mat_all[t] = np.append(mat_all[t], dist_mat_all[t][~lset[t], q_idx][:, None], axis=1)
        return lset
    #with query_cost
    def query2(self, n, query_type):
        lset = torch.tensor(self.labeled_set)
        time_span = torch.tensor(self.dataset.targets).shape[0]
        node_num = torch.tensor(self.dataset.targets).shape[1]
        qcost = []
        used_budget = 0
        mat_all = []
        dist_mat_all = []
        model = self.models[query_type].cuda()
        for time, snapshot in enumerate(self.dataset):
            x = snapshot.x.cuda()
            n_pool = x.shape[0]
            edge_index = snapshot.edge_index.cuda()
            edge_weight = snapshot.edge_attr.cuda()
            labeled_idxs = self.labeled_set[time]
            embeddings = model(x, edge_index, edge_weight).to('cpu')
            embeddings = embeddings.detach().numpy()
            dist_mat = np.matmul(embeddings, embeddings.transpose())
            sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
            dist_mat *= -2
            dist_mat += sq
            dist_mat += sq.transpose()
            dist_mat = np.sqrt(np.abs(dist_mat))
            # ~labeled_idxs 选择标FALSE的行表示未标注的节点，labeled_idxs选择标True的列表示已标注的节点
            # mat_all用来装非标记节点 到节点的距离，dis_mat_all放所有节点到所有节点的距离
            mat_all.append(dist_mat[~labeled_idxs, :][:, labeled_idxs])
            qcost.append(self.query_cost[time].numpy()[~labeled_idxs])
            dist_mat_all.append(dist_mat)
        qcost_min = min([np.array(i).min() for i in qcost])
        # mat_all = np.array(mat_all)
        # dist_mat_all = np.array(dist_mat_all)
        # 找出需要标记的点位置与时间
        for i in tqdm(range(n), ncols=100):
            t = 0
            # 返回各行的和
            l = -1
            #使用的预算加上qcost_min不能大于n
            if used_budget+qcost_min>n:
                break
            while l == -1:
                mat_min = [np.array(i).min(axis=1) for i in mat_all]
                # print('111')
                t = np.array([np.max(mat_min[i]/qcost[i]) for i in range(len(mat_min))]).argmax()
                #t = np.array([np.max(i) for i in mat_min]).argmax()
                # 局部最大值
                q_idx_ = np.argmax(mat_min[t])
                q_idx = np.arange(node_num)[~lset[t]][q_idx_]
                if mat_min[t][q_idx_] == -100:
                    break
                if used_budget + qcost[t][q_idx_]>n:
                    dist_mat_all[t][q_idx] = np.ones(dist_mat_all[t][q_idx].shape)*-100
                    mat_all[t][q_idx_] = np.ones(mat_all[t][q_idx_].shape)*-100
                    continue
                # 全局最大值

                # print('222')
                # 如果该时间选择节点超过了预算
                if sum(self.query_cost[t][lset[t]].tolist())+min(self.query_cost[t][~lset[t]].tolist())>self.budget:
                #if len(np.ones(node_num)[lset[t]].tolist()) >= self.budget:
                    # mat_min[q_idx_] = 0
                    dist_mat_all[t] = np.ones(dist_mat_all[t].shape)*-100
                    mat_all[t] = np.ones(mat_all[t].shape)*-100
                    # mat_all = dist_mat_all[~t_labeled_set]
                    continue
                # 如果选择q_idx会导致超过预算
                elif sum(self.query_cost[t][lset[t]].tolist())+self.query_cost[t][q_idx]>self.budget:
                    dist_mat_all[t][q_idx] = np.ones(dist_mat_all[t][q_idx].shape)*-100
                    mat_all[t][q_idx_] = np.ones(mat_all[t][q_idx_].shape)*-100
                    continue
                else:
                    # 没超过越算
                    lset[t][q_idx] = True
                    used_budget += self.query_cost[t][q_idx]
                    qcost[t] = self.query_cost[t].numpy()[~lset[t]]
                    l = 0
            mat_all[t] = dist_mat_all[t][~lset[t], :][:, lset[t]]
            #mat_all[t] = np.delete(mat_all[t], q_idx_, 0)
            #mat_all[t] = np.append(mat_all[t], dist_mat_all[t][~lset[t], q_idx][:, None], axis=1)
        return lset
    def query3(self, n, query_type):
        lset = torch.tensor(self.labeled_set)
        time_span = torch.tensor(self.dataset.targets).shape[0]
        node_num = torch.tensor(self.dataset.targets).shape[1]
        selected_ids = []
        qcost = []
        used_budget = 0
        mat_all = []
        dist_mat_all = []
        model = self.models[query_type].cuda()
        for time, snapshot in enumerate(self.dataset):
            x = snapshot.x.cuda()
            n_pool = x.shape[0]
            edge_index = snapshot.edge_index.cuda()
            edge_weight = snapshot.edge_attr.cuda()
            labeled_idxs = self.labeled_set[time]
            embeddings = model(x, edge_index, edge_weight).to('cpu')
            embeddings = embeddings.detach().numpy()
            dist_mat = np.matmul(embeddings, embeddings.transpose())
            sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
            dist_mat *= -2
            dist_mat += sq
            dist_mat += sq.transpose()
            dist_mat = np.sqrt(np.abs(dist_mat))
            # ~labeled_idxs 选择标FALSE的行表示未标注的节点，labeled_idxs选择标True的列表示已标注的节点
            # mat_all用来装非标记节点 到节点的距离，dis_mat_all放所有节点到所有节点的距离
            mat_all.append(dist_mat[~labeled_idxs, :][:, labeled_idxs])
            qcost.append(self.query_cost[time].numpy()[~labeled_idxs])
            dist_mat_all.append(dist_mat)
        qcost_min = min([np.array(i).min() for i in qcost])
        # mat_all = np.array(mat_all)
        # dist_mat_all = np.array(dist_mat_all)
        # 找出需要标记的点位置与时间
        for i in tqdm(range(n), ncols=100):
            t = 0
            # 返回各行的和
            l = -1
            #使用的预算加上qcost_min不能大于n
            if used_budget+qcost_min>n:
                break
            while l == -1:
                mat_min = [np.array(i).min(axis=1) for i in mat_all]
                # print('111')
                t = np.array([np.max(mat_min[i]/qcost[i]) for i in range(len(mat_min))]).argmax()
                #t = np.array([np.max(i) for i in mat_min]).argmax()
                # 局部最大值
                q_idx_ = np.argmax(mat_min[t])
                q_idx = np.arange(node_num)[~lset[t]][q_idx_]
                if mat_min[t][q_idx_] == -100:
                    break
                if used_budget + qcost[t][q_idx_]>n:
                    dist_mat_all[t][q_idx] = np.ones(dist_mat_all[t][q_idx].shape)*-100
                    mat_all[t][q_idx_] = np.ones(mat_all[t][q_idx_].shape)*-100
                    continue
                # 全局最大值

                # print('222')
                # 如果该时间选择节点超过了预算
                if sum(self.query_cost[t][lset[t]].tolist())+min(self.query_cost[t][~lset[t]].tolist())>self.budget:
                #if len(np.ones(node_num)[lset[t]].tolist()) >= self.budget:
                    # mat_min[q_idx_] = 0
                    dist_mat_all[t] = np.ones(dist_mat_all[t].shape)*-100
                    mat_all[t] = np.ones(mat_all[t].shape)*-100
                    # mat_all = dist_mat_all[~t_labeled_set]
                    continue
                # 如果选择q_idx会导致超过预算
                elif sum(self.query_cost[t][lset[t]].tolist())+self.query_cost[t][q_idx]>self.budget:
                    dist_mat_all[t][q_idx] = np.ones(dist_mat_all[t][q_idx].shape)*-100
                    mat_all[t][q_idx_] = np.ones(mat_all[t][q_idx_].shape)*-100
                    continue
                else:
                    # 没超过越算
                    lset[t][q_idx] = True
                    selected_ids.append(t*node_num+q_idx)
                    used_budget += self.query_cost[t][q_idx]
                    qcost[t] = self.query_cost[t].numpy()[~lset[t]]
                    l = 0
            mat_all[t] = dist_mat_all[t][~lset[t], :][:, lset[t]]
            #mat_all[t] = np.delete(mat_all[t], q_idx_, 0)
            #mat_all[t] = np.append(mat_all[t], dist_mat_all[t][~lset[t], q_idx][:, None], axis=1)
        return selected_ids





    '''

    def query(self, n, query_type): #query_type including GCN or RGCN
        # change
        lset = torch.tensor(self.labeled_set)
        t_labeled_set = lset.reshape(lset.shape[0]*lset.shape[1]).numpy()
        # standarded deviation std determines weights of model_free, gcn, tgcn
        # 假设所有时间都已经有了标识节点

        #alph = math.cos(math.pi*self.cycle/2*200)
        time_span = torch.tensor(self.dataset.targets).shape[0]
        node_num = torch.tensor(self.dataset.targets).shape[1]
        mat_all = []
        dist_mat_all = []
        for time, snapshot in enumerate(self.dataset):
            #print(self.dataset)
            #print(snapshot.x)
            #exit()
            x = snapshot.x
            n_pool = x.shape[0]
            edge_index = snapshot.edge_index
            edge_weight = snapshot.edge_attr
            labeled_idxs = self.labeled_set[time]
            # get model_free embeddings for calculating the
            #free_embeddings = self.models['Model_free'](x, edge_index, edge_weight)
            #free_embeddings = free_embeddings.detach().numpy()
            #gcn_embeddings = self.models['GCN'](x, edge_index, edge_weight)
            #gcn_embeddings = gcn_embeddings.detach().numpy()
            embeddings = self.models[query_type](x, edge_index, edge_weight)
            embeddings = embeddings.detach().numpy()
            dist_mat = np.matmul(embeddings, embeddings.transpose())
            sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
            dist_mat *= -2
            dist_mat += sq
            dist_mat += sq.transpose()
            dist_mat = np.sqrt(np.abs(dist_mat))
            #~labeled_idxs 选择标FALSE的行表示未标注的节点，labeled_idxs选择标True的列表示已标注的节点
            # mat_all用来装非标记节点 到节点的距离，dis_mat_all放所有节点到所有节点的距离
            mat_all.append(dist_mat[~labeled_idxs, :][:, labeled_idxs])
            dist_mat_all.append(dist_mat)
        mat_all = torch.tensor(mat_all)
        #print(mat_all.shape) (6, 6506, 100)
        mat_all = mat_all.reshape(mat_all.shape[0]*mat_all.shape[1],mat_all.shape[2]).numpy()
        dist_mat_all = torch.tensor(dist_mat_all)
        #print(dist_mat_all.shape) (6, 6606, 6606)
        dist_mat_all = dist_mat_all.reshape(dist_mat_all.shape[0] * dist_mat_all.shape[1], dist_mat_all.shape[2]).numpy()
        for i in tqdm(range(n), ncols=100):
            #返回各行的和
            l = -1
            while l == -1:
                mat_min = mat_all.min(axis=1)
                # 从未标注的各行和中选出最大的
                q_idx_ = mat_min.argmax()
                # 获取最大的节点所在位置
                q_idx = np.arange(time_span * node_num)[~t_labeled_set][q_idx_]
                #real_q_idx = q_idx % node_num
                # 标注为True
                start = q_idx // node_num * node_num
                end = start + node_num
                #如果超过了预算
                if len(np.ones(node_num)[t_labeled_set[start:end]].tolist())>=self.budget:
                    #mat_min[q_idx_] = 0
                    dist_mat_all[start:end] = torch.zeros(dist_mat_all[start:end].shape)
                    new_start = start - np.ones(time_span*node_num)[0:start][t_labeled_set[0:start]].shape[0]
                    new_end = end - np.ones(time_span*node_num)[0:end][t_labeled_set[0:end]].shape[0]
                    mat_all[new_start:new_end] = torch.zeros(mat_all[new_start:new_end].shape)
                    print(dist_mat_all[start:end])
                    print(mat_all[new_start:new_end])
                    #mat_all = dist_mat_all[~t_labeled_set]
                    continue
                else:
                    #没超过越算
                    t_labeled_set[q_idx] = True
                    l = 0
            #mat中删除
            print('wwww')
            print(mat_all.shape)
            mat_all = np.delete(mat_all, q_idx_, 0)
            print(mat_all.shape)
            print(dist_mat_all[~t_labeled_set, q_idx%node_num].shape)
            mat_all = np.append(mat_all, dist_mat_all[~t_labeled_set, q_idx%node_num][:, None], axis=1)
            print(mat_all.shape)
            #^为对比，如果相同就位FALSE，不同就为TRUE，通过该方式可以选出变化后的idxs
        t_labeled_set = torch.tensor(t_labeled_set.tolist())
        t_labeled_set = t_labeled_set.reshape(lset.shape).numpy()

        return t_labeled_set
            #np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]
        '''


'''
        one_list = np.ones(self.labeled_set.shape)
        labeled_num_list =  []
        for i in range(len(self.labeled_set)):
            try:
                labeled_num_list.append(len(one_list[i][self.labeled_set[i]]))
            except:
                labeled_num_list.append(0)
        std = np.std(labeled_num_list, ddof=1)
        '''
'''
        for j in range(self.labeled_set.shape[0]):
            self.labeled_set[j] = self.labeled_set[j] ^ t_labeled_set[j]
        '''