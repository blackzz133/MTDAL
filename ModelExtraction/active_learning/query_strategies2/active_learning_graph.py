import math
import numpy as np
import torch
#from kmeans_pytorch import kmeans
from tqdm import tqdm
class ALG_strategy:
    def __init__(self, dataset, models, labeled_set, cycle, all_loss, args, Rep):
        self.dataset = dataset
        self.models = models
        self.labeled_set = labeled_set
        self.cycle = cycle
        self.budget = args.budget
        self.all_loss = all_loss
        self.time_span = torch.tensor(labeled_set).shape[0]
        self.node_num = torch.tensor(labeled_set).shape[1]
        self.total_cycle = round(self.time_span * args.budget / args.queries)
        self.Rep = Rep  # 节点的representative

    def query(self, n):
        lset = torch.tensor(self.labeled_set)
        t_labeled_set = lset.reshape(lset.shape[0] * lset.shape[1]).numpy()

        alph = math.cos(math.pi * (self.cycle + 1) / (2 * self.total_cycle))
        beta = 0
        # imp: importance of nodes
        # 有问题
        # a = self.rep()
        a = self.Rep
        b = self.inf()
        # print('wwwwww')
        # print(a)
        # print(b)
        imp = alph * a + (1 - alph) * b
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
        # print('&&')
        # print(torch.max(imp))
        # print(torch.max(unlabeled_imp))
        # print(imp.shape)
        ##############
        for i in tqdm(range(n), ncols=100):
            # 返回各行的和
            l = -1
            while l == -1:
                q_idx_ = unlabeled_imp.argmax()

                # print('***')
                # print(q_idx_)
                # print(unlabeled_imp[q_idx_])

                # 获取最大的节点所在位置
                q_idx = np.arange(self.labeled_set.shape[0] * self.labeled_set.shape[1])[~t_labeled_set][q_idx_]

                # print(q_idx)
                # print(a[q_idx])
                # print(b[q_idx])
                # print(alph*a[q_idx])
                # print((1-alph)*b[q_idx])

                start = q_idx // self.node_num * self.node_num
                end = start + self.node_num
                if len(np.ones(self.node_num)[t_labeled_set[start:end]].tolist()) >= self.budget:
                    # unlabeled_imp[q_idx_] = 0
                    # 区间内所有imp值归0，以后不再选择
                    imp[start:end] = torch.tensor([0 for i in range(start, end)])
                    unlabeled_imp = imp[~t_labeled_set]
                    continue
                else:
                    t_labeled_set[q_idx] = True
                    l = 0
                    # unlabeled_imp中删除
                unlabeled_imp = np.delete(unlabeled_imp, q_idx_, 0)
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

        # std = np.std(labeled_num_list, ddof=1)

    # 一维和二维已经融合
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

    # 一维和二维已经融合
    def inf(self):
        all_type_embeddings = {}
        INF = []
        for key, value in self.models.items():
            all_embeddings = []
            if key != 'FREE' and key != 'victim':
                for time, snapshot in enumerate(self.dataset):
                    model = self.models[key].to('cpu')
                    embeddings = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr).detach()
                    # print(embeddings)
                    embeddings = embeddings.tolist()
                    all_embeddings.append(embeddings)
                all_embeddings = torch.tensor(all_embeddings)
                all_embeddings = all_embeddings.reshape(self.labeled_set.shape[0] * self.labeled_set.shape[1],
                                                        all_embeddings.shape[2])
                all_type_embeddings[key] = all_embeddings
        for i in range(self.labeled_set.shape[0] * self.labeled_set.shape[1]):
            result = 0
            for key, value in self.all_loss.items():
                for key2, value2 in self.all_loss.items():
                    error1 = 0.5 * torch.log2((1 - value) / value)
                    error2 = 0.5 * torch.log2((1 - value2) / value2)
                    result += error1 * error2 * torch.dist(all_type_embeddings[key][i], all_type_embeddings[key2][i])
            INF.append(result)

        return torch.tensor(INF)




