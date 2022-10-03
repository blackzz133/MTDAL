import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances

# 从a双层列表中除去b列表中的元素c=a-b
def double_list_minus(a, b):
    c = [[] for i in range(len(a))]
    try:
        for i in range(len(a)):
            c[i] = [x for x in a[i] if x not in b[i]]
        return c
    except:
        return c
#基于所有时间段节点indices生成可能的时序采样,前一个节点确定的时候，第二个节点需要为其领接节点
def temporal_sampling_genertator(indices, edge_index):

    time = len(indices)
    edge_index_t =np.array(torch.LongTensor(edge_index).transpose(0, 1)).tolist()
    samples = [[] for i in range(time-1)]
    node_num = len(indices[0])
    for i in range(time-1):
        for j in range(1,node_num+1):
            for k in range(1, node_num+1):
                if [j,k] in edge_index_t:
                    samples[i].append((j, k))
    return samples

def importance_by_order(features, labeled_set):
    importance = []
    return importance

def add_selected_node_indices(labeled_set, arg):
    for i in range(len(labeled_set)):
        for j in arg[i]:
            if arg[i][j] in labeled_set[i]:
                continue
            else:
                labeled_set[i].append(j)
    return labeled_set

def labeled_dataset_division(label):
    return

class SupervisedKMeans(ClassifierMixin, KMeans):
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centers_ = np.array([np.mean(X[y == c], axis=0) for c in self.classes])
        self.cluster_centers_ = self.centers_
        return self

    def predict(self, X):
        ed = euclidean_distances(X, self.cluster_centers_)
        return [self.classes[k] for k in np.argmin(ed, axis=1)]

    def score(self, X, y):
        y_ = self.predict(X)
        return np.mean(y == y_)


class SemiKMeans(SupervisedKMeans):
    def fit(self, X0, y0, X1):
        """To fit the semisupervised model

        Args:
            X0 (array): input variables with labels
            y0 (array): labels
            X1 (array): input variables without labels

        Returns:
            the model
        """
        classes0 = np.unique(y0)
        classes1 = np.setdiff1d(np.arange(self.n_clusters), classes0)
        self.classes = np.concatenate((classes0, classes1))

        X = np.row_stack((X0, X1))
        n1 = len(classes1)
        mu0 = SupervisedKMeans().fit(X0, y0).centers_
        if n1:
            centers, indices = kmeans_plusplus(X1, n_clusters=n1)
            self.cluster_centers_ = np.row_stack((centers, mu0))
        else:
            self.cluster_centers_ = mu0
        for _ in range(30):
            ED = euclidean_distances(X1, self.cluster_centers_)
            y1 = [self.classes[k] for k in np.argmin(ED, axis=1)]
            y = np.concatenate((y0, y1))
            self.cluster_centers_ = np.array([np.mean(X[y == c], axis=0) for c in self.classes])
        return self