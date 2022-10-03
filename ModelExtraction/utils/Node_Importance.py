import math
import torch
from torch import tensor
import numpy as np
import networkx as nx
G = nx.karate_club_graph()

degree_centrality = nx.degree_centrality(G)
centrality = nx.pagerank(G)
degree_centrality.values = degree_centrality.values()/sum(degree_centrality.values())
print(degree_centrality)
print(centrality.values())
exit()

#print()


class Node_Importance:
    def __init__(self, x=None, h=None, graphs=None):
        self.x = x
        self.h = h
        self.graphs = graphs
        self.epoch_time = 0
        self.importance_order = []
        self.importance = 0
        self.uncertainty = []
        self.representative = 0
        self.centrality = 0
        self.weight_alph = 0
        self.weight_beta = 0
        self.weight_gamma = 0
        self.node_num = x.shape[0]



    def weight_of_node_importance(self):
        weight = 0

    def uncertainty(self, prob_dist):
        #h值为节点数*特征数的维度
        '''
        for node_index in range(self.node_num):
            entropy = []
            for p in self.h[node_index]:
                entropy.append(-p*math.log2(p))
            self.uncertainty.append(sum(entropy))
        '''
        prob_dist = tensor([0.0321, 0.6439, 0.0871, 0.2369])
        log_probs = prob_dist * torch.log2(prob_dist)
        raw_entropy = 0 - torch.sum(log_probs)
        normalized_entropy = raw_entropy / math.log2(prob_dist.numel())
        return normalized_entropy

    def representative(self, x):
        self.representative = 0

    def centrality(self, x):
        degree_centrality = []
        degree_centrality = nx.degree_centrality(self.graph)
        self.centrality = 0

    def importance(self):
        self.importance = 0
        self.importance_order = []

    def importance_x(self, x):
        ordered_labeled_index = []
        node_num = torch.Tensor(x).shape[0]
        distance = torch.zeros(node_num)
        for i in range(0, node_num):
            a = list(np.arange(0, node_num, 1))
            a.remove(i)
            distance_to_othernode = []
            for j in a:
                distance_to_othernode.append(sum(abs((x[i] ** 2 - x[j] ** 2))) ** 0.5)
                if len(distance_to_othernode) == 2:
                    break
            distance[i] = sum(distance_to_othernode) / len(distance_to_othernode)
        for k in range(0, node_num):
            selected_node_index = torch.argmin(distance)
            ordered_labeled_index.append(selected_node_index)
            distance[selected_node_index] = 0
        return ordered_labeled_index

    def important_x_by_time(self, x):
        ordered_label_indexes = []
        total_time = torch.Tensor()


    def get_order(self):
        importance_order = []


