import torch
import numpy as np
import networkx as nx
from dataset_loader import DBLPELoader
import argparse
from victim import Chickenpox_victim_model
from attack import Chickenpox_attack_model
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
parser.add_argument("-e","--no_of_epochs", type=int, default=20,
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="first_query",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=5,
                    help="Number of active learning cycles")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-q","--queries", type=int, default=2,
                    help="Number of queries in a circle")
parser.add_argument("-b","--budget", type=int, default=4,
                    help="queried node budget at a snapshot")
parser.add_argument("-p","--span", type=int, default=10,
                    help="time span of a subquery")
parser.add_argument("-cn","--class_num", type=int, default=1,
                    help="the number of classes")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
#parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
#parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
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
    #url = 'features.txt'
    print('victim_model')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # victim_type = 'DCRNN' #DCRNN, EVOLVEGCNO, TGCN, A3TGCN
    victim_type = 'A3TGCN'
    victim_model = Chickenpox_victim_model(args, victim_type)
    #print(list(victim_model.parameters())[0].detach().shape)
    #print(list(victim_model.parameters())[1].detach().shape)
    #exit()
    #victim_model = None
    print('attack_model')
    attack_type = 'Random'#'Kcenter'#ALTG,  #Random MTTAL，M_Kcenter,
    #需要获取
    #attack_model, result, result2, result3, result4, gcn_result = Chickenpox_attack_model(args, victim_model, attack_type, device)
    #print('The f1_score result of rgcn is '+ str(result))
    #print('The f1_score result of rgcn2 is ' + str(result2))
    #print('The f1_score result of rgcn3 is ' + str(result3))
    #print('The f1_score result of rgcn4 is ' + str(result4))
    #print('The f1_score result of gcn is ' + str(gcn_result))
    attack_model, all_loss = Chickenpox_attack_model(args, victim_model, attack_type, device)


if __name__=='__main__':
    main()

