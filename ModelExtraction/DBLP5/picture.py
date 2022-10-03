import torch
import numpy as np
import networkx as nx
from dataset_loader import DBLPELoader,DBLPLoader
from TGCN.signal.train_test_split import temporal_signal_split, random_temporal_signal_split,node_data_sampling
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from sklearn.decomposition import PCA




loader = DBLPLoader('DBLP3')
dataset = loader.get_dataset()
#dataset= node_data_sampling(dataset, sampling_num=2000)
train_loader, test_loader = temporal_signal_split(dataset, 0.8)

nodecolor = ['red','blue','green','yellow','peru']
nodelabel =1
Node_class = len(dataset.targets[0][0])
plt.figure(figsize=(16,12))
print(Node_class)
print(len(np.unique(Node_class)))

for time, snapshot in enumerate(dataset):
    for time, snapshot in enumerate(dataset):
        x= snapshot.x
        labels = torch.argmax(snapshot.y, dim=1).tolist()
        colors =[nodecolor[i] for i in labels]
        data = Data(x=x, y=snapshot.y, edge_index = snapshot.edge_index, edge_attr = snapshot.edge_attr)
        g = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(g)
        #nx.draw(g,pos=pos, node_color = colors)
        nx.draw_networkx_edges(g, pos=pos, width=2, edge_color='black')
        nx.draw_networkx_nodes(g, pos=pos,node_color=colors,node_size=100)

        print('xxx')
        plt.show()
        exit()


