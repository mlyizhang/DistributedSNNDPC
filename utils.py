#! /usr/bin/env python
# -*- coding: utf-8 -*-

#############################
# utils files.
#############################
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

def load_dataset(filepath):
    """
        Return:
            dataset: dict
    """
    with open(filepath, 'rb') as fr:
        dataset = pickle.load(fr)
    return dataset
# 按行的方式计算两个坐标点之间的距离
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def SNN(k: int, nc: int, data: ndarray) -> Tuple[ndarray]:
	unassigned = -1
	n, d = data.shape

	# Compute distance
	# --------------------------------------------------------------------------------

	distance = squareform(pdist(data))

	# Compute neighbor
	# --------------------------------------------------------------------------------

	indexDistanceAsc: ndarray = argsort(distance)
	indexNeighbor: ndarray = indexDistanceAsc[:, :k]

	# Compute shared neighbor
	# --------------------------------------------------------------------------------

	indexSharedNeighbor = empty([n, n, k], int)
	numSharedNeighbor = empty([n, n], int)
	for i in range(n):
		numSharedNeighbor[i, i] = 0
		for j in range(i):
			shared: ndarray = intersect1d(indexNeighbor[i], indexNeighbor[j], assume_unique=True)
			numSharedNeighbor[j, i] = numSharedNeighbor[i, j] = shared.size
			indexSharedNeighbor[j, i, :shared.size] = indexSharedNeighbor[i, j, :shared.size] = shared

	# Compute similarity
	# --------------------------------------------------------------------------------

	similarity = zeros([n, n])  # Diagonal and some elements are 0
	for i in range(n):
		for j in range(i):
			if i in indexSharedNeighbor[i, j] and j in indexSharedNeighbor[i, j]:
				indexShared = indexSharedNeighbor[i, j, :numSharedNeighbor[i, j]]
				distanceSum = sum(distance[i, indexShared] + distance[j, indexShared])
				similarity[i, j] = similarity[j, i] = numSharedNeighbor[i, j] ** 2 / distanceSum

	# Compute ρ
	# --------------------------------------------------------------------------------

	rho = sum(sort(similarity)[:, -k:], axis=1)

	# Compute δ
	# --------------------------------------------------------------------------------

	distanceNeighborSum = empty(n)
	for i in range(n):
		distanceNeighborSum[i] = sum(distance[i, indexNeighbor[i]])
	indexRhoDesc = argsort(rho)[::-1]
	delta = full(n, inf)
	for i, a in enumerate(indexRhoDesc[1:], 1):
		for b in indexRhoDesc[:i]:
			delta[a] = min(delta[a], distance[a, b] * (distanceNeighborSum[a] + distanceNeighborSum[b]))
	delta[indexRhoDesc[0]] = -inf
	delta[indexRhoDesc[0]] = max(delta)
	gamma = rho * delta
	indexCentroid: ndarray = sort(argsort(gamma)[-nc:])
	return indexCentroid



def results(data_path):
    parameters=[]# number of centers in clients, number of centers in servers, k in SNN
    datapkl = load_dataset(data_path)
    #order = datapkl['order']
    data = list(datapkl['full_data'])
    corepoints = [] # save local kmeans centers.
    for i_client in range(datapkl['num_clusters']):
        # add noise to local data (Differential privacy)# 对中心点进行差分隐私保护,即为加上噪声。
        lodata = datapkl["client_" + str(i_client)]
        #nc = len(set(eachlable[i_client]))
        # n_clusters = min([len(lodata)//2,100])
        n_clusters =150
        cluster = KMeans(n_clusters).fit(lodata)
        m=cluster.cluster_centers_
        noise = np.random.laplace(0, 1/1, m.shape[0] * m.shape[1])
        noise = noise.reshape(m.shape[0], m.shape[1])
        m = m + noise

        corepoints.append(m)
    serverdata = np.concatenate(corepoints, axis=0)

    print(serverdata.nbytes)
    label = datapkl['true_label']
    parameters.append(n_clusters)
    cnum=len(set(label))
    #print('最终的类中心个数，',cnum)
    parameters.append(cnum)
    k= 25
    parameters.append(k)
    #print('snndpc中的参数k',k )
    centroid= SNN(k, cnum, serverdata)
    #finalcenter = cluster2.cluster_centers_
    finalcenter=[]
    ii=0
    for i in centroid:
        finalcenter.append(serverdata[i])
        ii=ii+1
    # 根据finalcenter分配所有客户端的数据点。
    idx = []

    finalcenter=np.array(finalcenter)

    print(finalcenter.nbytes)


    for i in data:
        simi = []
        for j in finalcenter:
            simi.append(np.linalg.norm(i - j))
        idx.append(simi.index(min(simi)) + 1)
    arr = np.array(idx)
    ari = round(adjusted_rand_score(label, arr),4 )
    nmi = round(normalized_mutual_info_score(label, arr),4)
    ami=  round(adjusted_mutual_info_score(label, arr),4)
    return ari, nmi,ami,parameters,arr

def fkmeans(data_path):
    parameters = []
    datapkl = load_dataset(data_path)  # dataset is a json file
    #eachlable = datapkl['eachlable']
    #order = datapkl['order']
    data = list(datapkl['full_data'])
    allgamme = np.zeros((1, len(data)))[0]
    corepoints = []
    client_results=[]
    for i_client in range(datapkl['num_clusters']):
        # add noise to local data (Differential privacy)
        lodata = datapkl["client_" + str(i_client)]
        noise = np.random.laplace(0, 1 / 2, lodata.shape[0] * lodata.shape[1])
        noise = noise.reshape(lodata.shape[0], lodata.shape[1])
        lodata = lodata + noise
        #nc = len(set(eachlable[i_client]))
        n_clusters = 200
        cluster = KMeans(n_clusters=n_clusters).fit(lodata)
        client_results.append(cluster.labels_)
        #print(cluster.cluster_centers_)
        ## 求局部密度和相对距离
        corepoints.append(cluster.cluster_centers_)
    serverdata = np.concatenate(corepoints, axis=0)

    nc=len(set(list(datapkl['true_label'])))
    cluster = KMeans(n_clusters=nc).fit(serverdata)
    finalcenter=cluster.cluster_centers_

    idx=np.zeros((1, len(data)))[0]
    label = datapkl['true_label']
    idx = []
    for i in data:
        simi = []
        for j in finalcenter:
            simi.append(np.linalg.norm(i - j))
        idx.append(simi.index(min(simi)) + 1)
    arr = np.array(idx)
    ari = round(adjusted_rand_score(label, arr), 4)
    nmi = round(normalized_mutual_info_score(label, arr), 4)
    ami = round(adjusted_mutual_info_score(label, arr), 4)
    parameters.append(n_clusters)
    parameters.append(nc)


    return ari, nmi, ami, parameters


import numpy as np


# 定义拉普拉斯噪声的函数
def add_laplace_noise(data, epsilon, sensitivity):
    """
    添加拉普拉斯噪声
    :param data: 原始数据
    :param epsilon: 隐私预算
    :param sensitivity: 数据的敏感性
    :return: 添加噪声后的数据
    """
    # 计算拉普拉斯分布的scale参数
    scale = sensitivity / epsilon
    # 生成与数据相同大小的拉普拉斯噪声
    noise = np.random.laplace(0, scale, data.shape)

    return data + noise


# 假设我们有以下原始数据
#data = np.array([10, 20, 30, 40, 50])

# 隐私预算和数据的敏感性
#epsilon = 1.0
#sensitivity = 1.0

# 添加拉普拉斯噪声
#noisy_data = add_laplace_noise(data, epsilon, sensitivity)
#print("Original Data: ", data)
#print("Noisy Data: ", noisy_data)