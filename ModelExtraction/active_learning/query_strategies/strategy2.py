import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Strategy2:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    def get_embeddings(self, data):
        x =1
        embeddings = self.net(data)
        return embeddings


