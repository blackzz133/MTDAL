from typing import Union, Tuple

from .static_graph_temporal_signal import StaticGraphTemporalSignal
from .dynamic_graph_temporal_signal import DynamicGraphTemporalSignal
from .dynamic_graph_static_signal import DynamicGraphStaticSignal

from .static_graph_temporal_signal_batch import StaticGraphTemporalSignalBatch
from .dynamic_graph_temporal_signal_batch import DynamicGraphTemporalSignalBatch
from .dynamic_graph_static_signal_batch import DynamicGraphStaticSignalBatch
import numpy as np
import torch
import random
from torch_geometric.utils.convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.utils.sparse import dense_to_sparse
import networkx as nx
import copy as cp
from scipy.sparse import coo_matrix
from scipy import sparse


Discrete_Signal = Union[StaticGraphTemporalSignal, StaticGraphTemporalSignalBatch,
                        DynamicGraphTemporalSignal, DynamicGraphTemporalSignalBatch,
                        DynamicGraphStaticSignal, DynamicGraphStaticSignalBatch]

def data_iter_consecutive(train_iterator, sampling_rate):
    time_span = torch.tensor(train_iterator.features).shape[0]
    num_steps = random.choice(range(round(time_span*sampling_rate), time_span+1))
    def _data(i):
        if type(train_iterator) == StaticGraphTemporalSignal:
            train_iterator1 = StaticGraphTemporalSignal(train_iterator.edge_index,
                                                        train_iterator.edge_weight,
                                                        train_iterator.features[i:i+num_steps],
                                                        train_iterator.targets[i:i+num_steps])

        elif type(train_iterator) == DynamicGraphTemporalSignal:

            train_iterator1 = DynamicGraphTemporalSignal(train_iterator.edge_indices[i:i+num_steps],
                                                         train_iterator.edge_weights[i:i+num_steps],
                                                         train_iterator.features[i:i+num_steps],
                                                         train_iterator.targets[i:i+num_steps])


        elif type(train_iterator) == DynamicGraphStaticSignal:
            train_iterator1 = DynamicGraphStaticSignal(train_iterator.edge_indices[i:i+num_steps],
                                                       train_iterator.edge_weights[i:i+num_steps],
                                                       train_iterator.feature,
                                                       train_iterator.targets[i:i+num_steps])

        if type(train_iterator) == StaticGraphTemporalSignalBatch:
            train_iterator1 = StaticGraphTemporalSignalBatch(train_iterator.edge_index,
                                                             train_iterator.edge_weight,
                                                             train_iterator.features[i:i+num_steps],
                                                             train_iterator.targets[i:i+num_steps],
                                                             train_iterator.batches)


        elif type(train_iterator) == DynamicGraphTemporalSignalBatch:
            train_iterator1 = DynamicGraphTemporalSignalBatch(train_iterator.edge_indices[i:i+num_steps],
                                                              train_iterator.edge_weights[i:i+num_steps],
                                                              train_iterator.features[i:i+num_steps],
                                                              train_iterator.targets[i:i+num_steps],
                                                              train_iterator.batches[i:i+num_steps])


        elif type(train_iterator) == DynamicGraphStaticSignalBatch:
            train_iterator1 = DynamicGraphStaticSignalBatch(train_iterator.edge_indices[i:i+num_steps],
                                                            train_iterator.edge_weights[i:i+num_steps],
                                                            train_iterator.feature,
                                                            train_iterator.targets[i:i+num_steps],
                                                            train_iterator.batches[i:i+num_steps])
        return train_iterator1

    samples = [_data(i) for i in range(time_span-num_steps+1)]

    return samples


def random_temporal_signal_split2(train_iterator, num_steps, batch_size):
    samples = []
    time = torch.tensor(train_iterator.features).shape[0]
    num_examples = (time -1)//num_steps
    example_indices = [i*num_steps for i in range(num_examples)]
    random.shuffle(example_indices)
    def _data(i):
        if type(train_iterator) == StaticGraphTemporalSignal:
            train_iterator1 = StaticGraphTemporalSignal(train_iterator.edge_index,
                                                        train_iterator.edge_weight,
                                                        train_iterator.features[i:i+num_steps],
                                                        train_iterator.targets[i:i+num_steps])

        elif type(train_iterator) == DynamicGraphTemporalSignal:

            train_iterator1 = DynamicGraphTemporalSignal(train_iterator.edge_indices[i:i+num_steps],
                                                         train_iterator.edge_weights[i:i+num_steps],
                                                         train_iterator.features[i:i+num_steps],
                                                         train_iterator.targets[i:i+num_steps])


        elif type(train_iterator) == DynamicGraphStaticSignal:
            train_iterator1 = DynamicGraphStaticSignal(train_iterator.edge_indices[i:i+num_steps],
                                                       train_iterator.edge_weights[i:i+num_steps],
                                                       train_iterator.feature,
                                                       train_iterator.targets[i:i+num_steps])

        if type(train_iterator) == StaticGraphTemporalSignalBatch:
            train_iterator1 = StaticGraphTemporalSignalBatch(train_iterator.edge_index,
                                                             train_iterator.edge_weight,
                                                             train_iterator.features[i:i+num_steps],
                                                             train_iterator.targets[i:i+num_steps],
                                                             train_iterator.batches)


        elif type(train_iterator) == DynamicGraphTemporalSignalBatch:
            train_iterator1 = DynamicGraphTemporalSignalBatch(train_iterator.edge_indices[i:i+num_steps],
                                                              train_iterator.edge_weights[i:i+num_steps],
                                                              train_iterator.features[i:i+num_steps],
                                                              train_iterator.targets[i:i+num_steps],
                                                              train_iterator.batches[i:i+num_steps])


        elif type(train_iterator) == DynamicGraphStaticSignalBatch:
            train_iterator1 = DynamicGraphStaticSignalBatch(train_iterator.edge_indices[i:i+num_steps],
                                                            train_iterator.edge_weights[i:i+num_steps],
                                                            train_iterator.feature,
                                                            train_iterator.targets[i:i+num_steps],
                                                            train_iterator.batches[i:i+num_steps])
        return train_iterator1
    for i in range(0, num_examples, batch_size):
        batch_indices = example_indices[i:i+batch_size]
        samples.append(_data(i))
    return samples





def random_temporal_signal_split(train_iterator)->Discrete_Signal:
    print(0,round(train_iterator.snapshot_count/2-1))
    start = np.random.randint(0, round(train_iterator.snapshot_count/2-1))
    print(start+1,train_iterator.snapshot_count-1)
    end = np.random.randint(start+1, train_iterator.snapshot_count-1)
    if type(train_iterator) == StaticGraphTemporalSignal:
        train_iterator1 = StaticGraphTemporalSignal(train_iterator.edge_index,
                                                   train_iterator.edge_weight,
                                                   train_iterator.features[start:end],
                                                   train_iterator.targets[start:end])

    elif type(train_iterator) == DynamicGraphTemporalSignal:

        train_iterator1 = DynamicGraphTemporalSignal(train_iterator.edge_indices[start:end],
                                                    train_iterator.edge_weights[start:end],
                                                    train_iterator.features[start:end],
                                                    train_iterator.targets[start:end])


    elif type(train_iterator) == DynamicGraphStaticSignal:
        train_iterator1 = DynamicGraphStaticSignal(train_iterator.edge_indices[start:end],
                                                  train_iterator.edge_weights[start:end],
                                                  train_iterator.feature,
                                                  train_iterator.targets[start:end])


    if type(train_iterator) == StaticGraphTemporalSignalBatch:
        train_iterator1 = StaticGraphTemporalSignalBatch(train_iterator.edge_index,
                                                        train_iterator.edge_weight,
                                                        train_iterator.features[start:end],
                                                        train_iterator.targets[start:end],
                                                        train_iterator.batches)


    elif type(train_iterator) == DynamicGraphTemporalSignalBatch:
        train_iterator1 = DynamicGraphTemporalSignalBatch(train_iterator.edge_indices[start:end],
                                                         train_iterator.edge_weights[start:end],
                                                         train_iterator.features[start:end],
                                                         train_iterator.targets[start:end],
                                                         train_iterator.batches[start:end])


    elif type(train_iterator) == DynamicGraphStaticSignalBatch:
        train_iterator1 = DynamicGraphStaticSignalBatch(train_iterator.edge_indices[start:end],
                                                       train_iterator.edge_weights[start:end],
                                                       train_iterator.feature,
                                                       train_iterator.targets[start:end],
                                                       train_iterator.batches[start:end])
    return train_iterator1



def temporal_signal_split(data_iterator, train_ratio: float=0.8) -> Tuple[Discrete_Signal, Discrete_Signal]:
    r""" Function to split a data iterator according to a fixed ratio.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **train_ratio** *(float)* - Graph edge indices.

    Return types:
        * **(train_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train and test data iterators.
    """
    train_snapshots = int(train_ratio*data_iterator.snapshot_count)
    #val_snapshots = int(train_snapshots+(1-train_ratio)/2*data_iterator.snapshot_count)
    #print(train_snapshots)
    if type(data_iterator) == StaticGraphTemporalSignal:
        train_iterator = StaticGraphTemporalSignal(data_iterator.edge_index,
                                                   data_iterator.edge_weight,
                                                   data_iterator.features[0:train_snapshots],
                                                   data_iterator.targets[0:train_snapshots])

        test_iterator = StaticGraphTemporalSignal(data_iterator.edge_index,
                                                  data_iterator.edge_weight,
                                                  data_iterator.features[train_snapshots:],
                                                  data_iterator.targets[train_snapshots:])

    elif type(data_iterator) == DynamicGraphTemporalSignal:

        train_iterator = DynamicGraphTemporalSignal(data_iterator.edge_indices[0:train_snapshots],
                                                    data_iterator.edge_weights[0:train_snapshots],
                                                    data_iterator.features[0:train_snapshots],
                                                    data_iterator.targets[0:train_snapshots])

        test_iterator = DynamicGraphTemporalSignal(data_iterator.edge_indices[train_snapshots:],
                                                   data_iterator.edge_weights[train_snapshots:],
                                                   data_iterator.features[train_snapshots:],
                                                   data_iterator.targets[train_snapshots:])
                                                   
    elif type(data_iterator) == DynamicGraphStaticSignal:
        train_iterator = DynamicGraphStaticSignal(data_iterator.edge_indices[0:train_snapshots],
                                                  data_iterator.edge_weights[0:train_snapshots],
                                                  data_iterator.feature,
                                                  data_iterator.targets[0:train_snapshots])

        test_iterator = DynamicGraphStaticSignal(data_iterator.edge_indices[train_snapshots:],
                                                 data_iterator.edge_weights[train_snapshots:],
                                                 data_iterator.feature,
                                                 data_iterator.targets[train_snapshots:])
                                                 
    if type(data_iterator) == StaticGraphTemporalSignalBatch:
        train_iterator = StaticGraphTemporalSignalBatch(data_iterator.edge_index,
                                                        data_iterator.edge_weight,
                                                        data_iterator.features[0:train_snapshots],
                                                        data_iterator.targets[0:train_snapshots],
                                                        data_iterator.batches)

        test_iterator = StaticGraphTemporalSignalBatch(data_iterator.edge_index,
                                                       data_iterator.edge_weight,
                                                       data_iterator.features[train_snapshots:],
                                                       data_iterator.targets[train_snapshots:],
                                                       data_iterator.batches)

    elif type(data_iterator) == DynamicGraphTemporalSignalBatch:
        train_iterator = DynamicGraphTemporalSignalBatch(data_iterator.edge_indices[0:train_snapshots],
                                                         data_iterator.edge_weights[0:train_snapshots],
                                                         data_iterator.features[0:train_snapshots],
                                                         data_iterator.targets[0:train_snapshots],
                                                         data_iterator.batches[0:train_snapshots])

        test_iterator = DynamicGraphTemporalSignalBatch(data_iterator.edge_indices[train_snapshots:],
                                                        data_iterator.edge_weights[train_snapshots:],
                                                        data_iterator.features[train_snapshots:],
                                                        data_iterator.targets[train_snapshots:],
                                                        data_iterator.batches[train_snapshots:])
                                                   
    elif type(data_iterator) == DynamicGraphStaticSignalBatch:
        train_iterator = DynamicGraphStaticSignalBatch(data_iterator.edge_indices[0:train_snapshots],
                                                       data_iterator.edge_weights[0:train_snapshots],
                                                       data_iterator.feature,
                                                       data_iterator.targets[0:train_snapshots],
                                                       data_iterator.batches[0:train_snapshots:])

        test_iterator = DynamicGraphStaticSignalBatch(data_iterator.edge_indices[train_snapshots:],
                                                      data_iterator.edge_weights[train_snapshots:],
                                                      data_iterator.feature,
                                                      data_iterator.targets[train_snapshots:],
                                                      data_iterator.batches[train_snapshots:])

    return train_iterator, test_iterator

def node_data_sampling(data_iterator, sampling_num):
    '''

    node_num = torch.tensor(dataset).shape[1]
    a = [i for i in range(node_num)]
    random.shuffle(a)
    selected_node_num = a[0:sampling_num]
    '''
    #random.seed(1)
    node_num = torch.tensor(data_iterator.features).shape[1]
    a = [i for i in range(node_num)]
    #random.shuffle(a)
    selected_node_nums = a[0:sampling_num]
    features = []
    targets = []
    edge_indices = []
    edge_weights = []
    for time, snapshot in enumerate(data_iterator):
        features.append(snapshot.x[selected_node_nums].numpy())
        targets.append(snapshot.y[selected_node_nums].numpy())
        matrix = to_scipy_sparse_matrix(snapshot.edge_index, snapshot.edge_attr, node_num)
        dense_matrix = coo_matrix(matrix).todense()
        dense_list = np.array(dense_matrix.tolist())[selected_node_nums][:, selected_node_nums]
        #print(dense_list.shape)
        #new_dense_matrix = np.matrix(dense_list)
        #print(new_dense_matrix)
        sub_matrix = sparse.csr_matrix(dense_list)
        #print(sub_matrix)
        #edge_index, edge_weight = dense_to_sparse(dense_list)
        edge_index, edge_weight = from_scipy_sparse_matrix(sub_matrix)
        edge_indices.append(edge_index.numpy())
        edge_weights.append(edge_weight.numpy())
    new_iterator = DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)
    return new_iterator


def node_data_sampling2(data_iterator, time_span):
    '''

        node_num = torch.tensor(dataset).shape[1]
        a = [i for i in range(node_num)]
        random.shuffle(a)
        selected_node_num = a[0:sampling_num]
        '''
    # random.seed(1)
    node_num = torch.tensor(data_iterator.features).shape[1]
    a = [i for i in range(node_num)]
    # random.shuffle(a)
    selected_time_span = a[0:time_span]
    features = []
    targets = []
    edge_indices = []
    edge_weights = []
    w = 0
    for time, snapshot in enumerate(data_iterator):
        w += 1
        features.append(snapshot.x.numpy())
        targets.append(snapshot.y.numpy())
        matrix = to_scipy_sparse_matrix(snapshot.edge_index, snapshot.edge_attr, node_num)
        dense_matrix = coo_matrix(matrix).todense()
        dense_list = np.array(dense_matrix.tolist())
        # print(dense_list.shape)
        # new_dense_matrix = np.matrix(dense_list)
        # print(new_dense_matrix)
        sub_matrix = sparse.csr_matrix(dense_list)
        # print(sub_matrix)
        # edge_index, edge_weight = dense_to_sparse(dense_list)
        edge_index, edge_weight = from_scipy_sparse_matrix(sub_matrix)
        edge_indices.append(edge_index.numpy())
        edge_weights.append(edge_weight.numpy())
        if w >= time_span:
            new_iterator = DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)
            return new_iterator
        else:
            continue






