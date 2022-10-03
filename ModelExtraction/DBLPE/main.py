import torch
import numpy as np
import networkx as nx
from dataset_loader import DBLPELoader
import argparse
from victim import DBLPE_victim_model
from attack import DBLP_attack_model
from sklearn.metrics import f1_score
from TGCN.signal.train_test_split import temporal_signal_split
import os
from config import *


parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=1.2,
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="chickenpox",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=5,
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="first_query",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=5,
                    help="Number of active learning cycles")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-q","--queries", type=int, default=10,
                    help="Number of queries in a circle")
parser.add_argument("-b","--budget", type=int, default=20,
                    help="queried node budget at a snapshot")
parser.add_argument("-p","--span", type=int, default=2,
                    help="time span of a subquery")
parser.add_argument("-k","--sampling_rate", type=float, default=0.5,
                    help="time span of a subquery")
parser.add_argument("-cn","--class_num", type=int, default=2,
                    help="the number of classes")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
#parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
#parser.add_argument('--n_round', type=int, 赵梦玥default=10, help="number of rounds")
#parser.add_argument('--dataset_name', type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="RandomSampling",
                    choices=["RandomSampling",
                             "LeastConfidence",
                             "MarginSampling",
                             "EntropySampling",
                             "LeastConfidenceDropout",
                             "MarginSamplingDropout",
                             "EntropySamplingDropout",
                             "KMeansSampling",
                             "KCenterGreedy",
                             "BALDDropout",
                             "AdversarialBIM",
                             "AdversarialDeepFool"], help="query strategy")
parser.add_argument('--method', type=str, default="Random",
                    choices=["Random",
                             "GCNPrior",
                             "RNNPrior",
                             "GCNOnly",
                             "first_query"], help="method")



def main():
    print('start')
    args = parser.parse_args()
    url = 'features.txt'
    print('victim_model')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    victim_type = 'DCRNN'  # DCRNN, EVOLVEGCNO, TGCN, A3TGCN
    victim_model = DBLPE_victim_model(args, victim_type)
    '''
    del victim_model
    print('**********')
    victim_type = 'EVOLVEGCNO'  # DCRNN, EVOLVEGCNO, TGCN, A3TGCN
    victim_model = DBLPE_victim_model(args, victim_type)
    del victim_model
    print('**********')
    victim_type = 'TGCN'  # DCRNN, EVOLVEGCNO, TGCN, A3TGCN
    victim_model = DBLPE_victim_model(args, victim_type)
    del victim_model
    print('**********')
    victim_type = 'A3TGCN'  # DCRNN, EVOLVEGCNO, TGCN, A3TGCN
    victim_model = DBLPE_victim_model(args, victim_type)
    print('**********')
    exit()
    '''
    #exit()
    #victim_model = None
    print('attack_model')
    dataset = 'DBLP5'
    #attack_type = 'ALTG'#ALG, DEAL+Kcenter
    #attack_type = 'ALTG'
    attack_type='MTTAL'
    #attack_type = 'Kcenter'
    #attack_type='Kcenter'
    print(attack_type)
    #需要获取
    models, all_loss = DBLP_attack_model(args, victim_model, dataset, attack_type, device)

    '''
    # victim_model testing is start
    loader = DBLPELoader()
    dataset = loader.get_dataset()
    train_loader, test_loader = temporal_signal_split(dataset, 0.8)
    result = 0
    total_time = 0
    for time, snapshot in enumerate(dataset):
        with torch.cuda.device(device=device):
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = snapshot.edge_attr.to(device)
            victim_labels = torch.argmax(victim_model.to(device)(x, edge_index, edge_attr).detach(),
                                         dim=1).long().clone().to('cpu')
            y = snapshot.y
            y = y.numpy()
            y = np.argmax(y, axis=1)
            labels = torch.from_numpy(y).long().cpu()

            result = result + (torch.eq(labels, victim_labels).sum() / labels.shape[0])
            total_time += 1
    print('The accuracy of victim_model is ' + str(result / total_time))
    exit()
    # victim testing is end
    '''



if __name__=='__main__':
    main()

