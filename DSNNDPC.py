from numpy import loadtxt, ndarray, min, max
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from numpy import *
from SNNDPC import SNNDPC
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import pickle
import math
from utils import *
import os
from sklearn.cluster import DBSCAN
import random
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from typing import List, Tuple
from numpy import arange, argsort, argwhere, empty, full, inf, intersect1d, max, ndarray, sort, sum, zeros
from scipy.spatial.distance import pdist, squareform
import sys
from pympler import asizeof


dataname=['iris','breast','ecoli','zoo','thyroid','wine','seeds','abalone','heart','waveform',
          'gesture','liver','ionosphere']

for i in dataname:
    data_path = '../../dataset/' + i + 'fed.pkl'
    datapkl = load_dataset(data_path)  # dataset is a json file
    label = datapkl['true_label']
    print("Processing dataset:", data_path)
    print(datapkl['full_data'].shape)
    parameters = []  # number of centers in clients, number of centers in servers, k in SNN
    datapkl = load_dataset(data_path)
    # order = datapkl['order']
    data = list(datapkl['full_data'])
    corepoints = []
    for i_client in range(datapkl['num_clusters']):
        lodata = datapkl["client_" + str(i_client)]
        n_clusters = min([len(lodata) // 3, 200])
        parameters.append(n_clusters)
        k1=5
        centroid, assignment = SNNDPC(k1, n_clusters, lodata)
        for i in centroid:
            corepoints.append(lodata[i].tolist())

    cnum = len(set(label))

    corepoints=np.array(corepoints)
    # serverdata = np.concatenate(corepoints, axis=0)
    # print(serverdata)
    parameters.append(cnum)
    k2=5
    final=[]
    finalcenter, assignment = SNNDPC(k2, cnum, corepoints)
    for i in finalcenter:
        final.append(corepoints[i].tolist())
    print(final)
    idx = []
    for i in data:
        simi = []
        for j in final:
            simi.append(np.linalg.norm(i - j))
        idx.append(simi.index(min(simi)) + 1)
    arr = np.array(idx)
    ari = round(adjusted_rand_score(label, arr), 4)
    nmi = round(normalized_mutual_info_score(label, arr), 4)
    print('number of centers in clients', parameters[0], 'number of centers in servers', parameters[1], 'k in knn',
          parameters[2])
    with open('result.txt', 'a') as f:
        f.write(data_path)
        f.write('ari' + str(ari) + 'nmi' + str(nmi) + '\n')
        # f.write('number of centers in clients' + str(parameters[0]) + 'number of centers in servers' +
        #         str(parameters[1]) + 'k in knn' + str(parameters[2]) + '\n')

