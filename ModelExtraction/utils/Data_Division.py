from TGCN.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from torch_geometric.data.data import Data
import torch

class Train_Data_Division:
    def __init__(self, train_datasets, labeled_indexes, signal_type):
        self.train_datasets = train_datasets
        self.labeled_indexes = labeled_indexes
        self.signal_type = signal_type

    def data_division(self):
        labeled_dataset = StaticGraphTemporalSignal(edge_index=self.train_datasets.edge_index,
                                                    edge_weight=self.train_datasets.edge_weight
                                                    , features=self.train_datasets.features, targets=self.train_datasets.targets)

        unlabeled_dataset = StaticGraphTemporalSignal(edge_index=self.train_datasets.edge_index,
                                                    edge_weight=self.train_datasets.edge_weight
                                                    , features=self.train_datasets.features, targets=self.train_datasets.targets)

        return labeled_dataset, unlabeled_dataset

class Snapshot_Data_Division:
    def __init__(self, snapshot, labeled_index):
        self.snapshot = snapshot
        self.labeled_index = labeled_index

    def divided_snapshot(self):
        node_num = self.snapshot.x.shape[0]
        feature_num = self.snapshot.x.shape[1]
        labeled_x = self.snapshot.x
        labeled_y = self.snapshot.y
        unlabeled_x = self.snapshot.x
        for i in range(0, node_num):
            if i not in self.labeled_index:
                labeled_x[i] == torch.zeros(labeled_x[i].shape)
                labeled_y[i] = torch.zeros(labeled_y[i].shape)
            else:
                unlabeled_x[i] = torch.zeros(unlabeled_x[i].shape)
        labeled_snapshot = Data(edge_index=self.snapshot.edge_index, edge_attr=self.snapshot.edge_attr, x=labeled_x, y=labeled_y)
        unlabeled_snapshot = Data(edge_index=self.snapshot.edge_index, edge_attr=self.snapshot.edge_attr, x=unlabeled_x, y=None)
        return labeled_snapshot, unlabeled_snapshot