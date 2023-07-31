
from __future__ import division
import os
import logging
from neuron import gui,h
from neuron.units import ms, mV
import numpy as np
# import neuron_reduce
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

# logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

#Create a L5_PC model (channels in biophys3.hoc and morphology in template.hoc)
h.load_file("import3d.hoc")
h.nrn_load_dll('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/mod/nrnmech.dll')
h.load_file('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/modelFile/L5PCbiophys3.hoc')
h.load_file('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/modelFile/L5PCtemplate.hoc')
complex_cell = h.L5PCtemplate('C:/Users/Windows/Desktop/Drew/neuron_reduce/example/modelFile/cell1.asc')
h.celsius = 37
h.v_init = complex_cell.soma[0].e_pas

import networkx as nx
import pandas as pd

#Add synapses to the model
synapses_list, netstims_list, netcons_list, randoms_list = [], [], [] ,[]

#Add segments of apical and basal sections for synapses (no soma and axon)
all_segments = [i for j in map(list,list(complex_cell.basal)) for i in j] + [i for j in map(list,list(complex_cell.apical)) for i in j] 
len_per_segment = np.array([seg.sec.L/seg.sec.nseg for seg in all_segments])

all_sections_soma = [i for i in map(list,list(complex_cell.soma))] 
all_sections_basal = [i for i in map(list,list(complex_cell.basal))] 
all_sections_apical = [i for i in map(list,list(complex_cell.apical))]
sections_basal_apical = all_sections_basal + all_sections_apical
all_sections = all_sections_soma + all_sections_basal + all_sections_apical

# Section_list = []
parentID_list, sectionID_list, sectionName_list, length_list = [], [], [], []

parent_list = []
parent_index_list = []
rnd = np.random.RandomState(10)

if rnd.uniform() < 0.85:
    e_syn, tau1, tau2, spike_interval, syn_weight = 0, 0.3, 1.8, 1000/2.5, 0.0016
else:
    e_syn, tau1, tau2, spike_interval, syn_weight = -86, 1, 8, 1000/15.0, 0.0008

df = pd.DataFrame(columns=['sectionID', 'parentID', 'sectionName', 'parentName','length'])

import re
i = 0
for section in all_sections:
    Section = section[0].sec
    sectionID = i
    sectionName = Section.psection()['name']
    match = match = re.search(r'\.(.*?)\[', sectionName)
    sectionType = match.group(1)
    
    L = Section.psection()['morphology']['L']

    # Section_list.append(Section)

    parent_list.append(sectionName)
    parent_index_list.append(sectionID)

    if i == 0:
        parentID = 0
        parentName = 'None'
    else:
        parent = Section.psection()['morphology']['parent'].sec
        parentName = parent.psection()['name']
        parentID = parent_index_list[parent_list.index(parentName)]

    sectionID_list.append(sectionID)
    sectionName_list.append(sectionName)
    parentID_list.append(parentID)
    length_list.append(L)
    
    # create data
    data = {'sectionID': sectionID, 
            'parentID': parentID, 
            'sectionName': sectionName,
            'parentName':parentName,
            # 'type':rnd.choice(['A','B']),
            'length': L,
            'sectionType': sectionType}

    df = pd.concat([df, pd.DataFrame(data,index=[0])])
    i = i + 1
df.to_csv("cell1.csv", encoding='utf-8', index=False)

# Create the graph based on the connectivity in the L5PC model
Data = open('cell1.csv', "r")
next(Data, None)  # skip the first line in the input file

Graphtype = nx.Graph()
G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                      nodetype=int, data=(('sectionName',str),('parentName',str),
                                          ('length',float),('sectionType',str)))
color_map = [''] * len(G.edges())
for edge in G.edges(data=True):
    if edge[2]['sectionType'] == 'soma':
        color_map[edge[1]] ='red' 
    elif edge[2]['sectionType'] == 'dend':
        color_map[edge[1]] ='blue' 
    elif edge[2]['sectionType'] == 'apic': 
        color_map[edge[1]] ='green' 
    
sp = dict(nx.all_pairs_shortest_path(G))

## Initialize synapses
numSyn = 1000
type_list = []
synapses_list = []
sectionID_synapse_list = []
loc_list = []

for i in tqdm(range(numSyn)):
        
    Section = rnd.choice(sections_basal_apical)
    section = Section[0].sec
    sectionName = section.psection()['name']
    sectionID_synapse = sectionID_list[sectionName_list.index(sectionName)]
    
    sectionID_synapse_list.append(sectionID_synapse)
    # Use to differentiate between input type A and B
    type_list.append(rnd.choice(['A','B']))

    loc = section(rnd.uniform()).x 
    loc_list.append(loc)
    synapses_list.append(h.Exp2Syn(section(loc)))

    synapses_list[i].e, synapses_list[i].tau1, synapses_list[i].tau2 = e_syn, tau1, tau2

    netstims_list.append(h.NetStim())
    netstims_list[i].interval, netstims_list[i].number, netstims_list[i].start, netstims_list[i].noise = spike_interval, 10, 100, 1

    randoms_list.append(h.Random())
    randoms_list[i].Random123(i)
    randoms_list[i].negexp(1)
    netstims_list[i].noiseFromRandom(randoms_list[i])

    netcons_list.append(h.NetCon(netstims_list[i], synapses_list[i])) # need to rewrite with an assign function
    netcons_list[i].delay, netcons_list[i].weight[0] = 0, syn_weight

    time.sleep(0.01)

# Calculate the distance between all pairs of synapses 
distance_list = []
distance_limit = 2000 #microns
distance_matrix = np.zeros((numSyn,numSyn))
for i in range(numSyn):
    for j in range(numSyn):
        if i < j:
            m = sectionID_synapse_list[i]
            n = sectionID_synapse_list[j]

            path = sp[m][n]
            
            distance = 0

            if len(path) > 1:
                loc_i = loc_list[i] * (parentID_list[m] == path[1]) + (1-loc_list[i]) * (parentID_list[m] != path[1])
                loc_j = loc_list[j] * (parentID_list[n] == path[-2]) + (1-loc_list[j]) * (parentID_list[n] != path[-2])
                for k in path:
                    if k == m:
                        distance = distance + length_list[k]*loc_i
                    if k == n:
                        distance = distance + length_list[k]*loc_j
                    distance = distance + length_list[k]
            else:
                distance = length_list[m] * abs(loc_list[i] - loc_list[j])
            
            distance_matrix[i,j] = distance_matrix[j,i] = distance
            distance_list.append(distance) #if distance < distance_limit else None
            
## Distance analysis
bin_list = [0, 2.7, 4.5, 7.4, 12, 20, 33, 55, 90, 148, 245]

## Cluster type
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

# this is not usable
def custom_spectral_clustering(distance_matrix, n_clusters, n_A, n_B):
    # 使用谱聚类将数据点分成两类
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=0)
    labels = spectral_clustering.fit_predict(distance_matrix)

    # 计算初始聚类结果中每个类别的点个数
    n_cluster_A = np.sum(labels == 0)
    n_cluster_B = np.sum(labels == 1)

    # 调整聚类结果，直到满足点个数约束
    while n_cluster_A != n_A or n_cluster_B != n_B:
        if n_cluster_A > n_A:
            # 如果A的个数多于n_A，我们从A类别中移动一些点到B类别中
            indices_to_move = np.where(labels == 0)[0][:n_cluster_A - n_A]
            labels[indices_to_move] = 1

        elif n_cluster_A < n_A:
            # 如果A的个数少于n_A，我们从B类别中移动一些点到A类别中
            indices_to_move = np.where(labels == 1)[0][:n_A - n_cluster_A]
            labels[indices_to_move] = 0

        # 更新每个类别的点个数
        n_cluster_A = np.sum(labels == 0)
        n_cluster_B = np.sum(labels == 1)

    return labels

def custom_kmeans(distance_matrix, n_clusters, n_A, n_B, max_iters=100000):
    # 转换距离矩阵为相似性矩阵（通过取倒数）
    similarity_matrix = 1 / (1 + distance_matrix)

    # 初始化聚类中心（可以随机选择n_clusters个点作为初始中心）
    center_indices = np.random.choice(len(similarity_matrix), size=n_clusters, replace=False)
    centers = similarity_matrix[center_indices]

    n_cluster_wished = np.array([n_A,  n_B])
    # 进行KMeans迭代更新
    for _ in range(max_iters):
        # 计算每个点到聚类中心的距离
        distances = np.linalg.norm(centers[:, np.newaxis] - similarity_matrix, axis=2)

        # 确定每个点的聚类标签
        # In this step, all labels of points are reassigned, 
        # so -1 we got in last iteration will be overwritten
        labels = np.argmin(distances, axis=0)

        n_cluster_prev = np.array([np.sum(labels == i) for i in range(n_clusters)])

        # 根据指定的n_A和n_B来调整聚类中心
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            # n_cluster_A = np.sum(labels[cluster_indices] == 0)
            # n_cluster_B = len(cluster_indices) - n_cluster_A

            if n_cluster_prev[i] > n_cluster_wished[i]:
                # 如果A的个数多于n_A，我们删除一些A类型的点
                indices_to_remove = np.random.choice(cluster_indices, size=n_cluster_prev[i] - n_cluster_wished[i], replace=False)
                labels[indices_to_remove] = -1  # 将删除的点标记为-1

            # if n_cluster_A > n_A:
            #     # 如果A的个数多于n_A，我们删除一些A类型的点
            #     indices_to_remove = np.random.choice(cluster_indices[labels[cluster_indices] == 0], size=n_cluster_A - n_A, replace=False)
            #     labels[indices_to_remove] = -1  # 将删除的点标记为-1

            # if n_cluster_B > n_B:
            #     # 如果B的个数多于n_B，我们删除一些B类型的点
            #     indices_to_remove = np.random.choice(cluster_indices[labels[cluster_indices] == 1], size=n_cluster_B - n_B, replace=False)
            #     labels[indices_to_remove] = -1  # 将删除的点标记为-1

            # 更新聚类中心
            cluster_indices = np.where(labels == i)[0]
            centers[i] = np.mean(similarity_matrix[cluster_indices], axis=0)

        # 如果没有点被删除，说明已经满足要求，停止迭代
        if np.sum(labels == -1) == 0:
            break
        
    # 删除被标记为-1的点
    valid_indices = np.where(labels != -1)[0]
    labels = labels[valid_indices]

    return labels

def custom_kmeans_1(distance_matrix, n_clusters, n_A, n_B, max_iters=10000):
    # 转换距离矩阵为相似性矩阵（通过取倒数）
    similarity_matrix = 1 / (1 + distance_matrix)

    # 初始化聚类中心（可以随机选择n_clusters个点作为初始中心）
    center_indices = np.random.choice(len(similarity_matrix), size=n_clusters, replace=False)
    centers = similarity_matrix[center_indices]

    n_cluster_wished = np.array([n_A,  n_B])
    for _ in range(max_iters):
        # 计算每个点到聚类中心的距离
        distances = np.linalg.norm(centers[:, np.newaxis] - similarity_matrix, axis=2)

        # 确定每个点的聚类标签
        labels = np.argmin(distances, axis=0)

        # 记录当前每个cluster的A类型点个数
        n_cluster_prev = np.array([np.sum(labels == i) for i in range(n_clusters)])

        # 调整聚类中心
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]

            if n_cluster_prev[i] > n_cluster_wished[i]:
                # 如果A的个数多于n_A，我们删除一些A类型的点
                indices_to_remove = np.random.choice(cluster_indices, size=int((n_cluster_prev[i] - n_cluster_wished[i])/2), replace=False)
                labels[indices_to_remove] = -1  # 将删除的点标记为-1

            # 更新聚类中心
            cluster_indices = np.where(labels == i)[0]
            centers[i] = np.mean(similarity_matrix[cluster_indices], axis=0)

        # 记录每个cluster的A类型点个数
        n_cluster_after = np.array([np.sum(labels == i) for i in range(n_clusters)])

        # 检查每个cluster的A类型点个数是否小于n_A
        for i in range(n_clusters):
            if n_cluster_after[i] < n_cluster_wished[i]:
                # 如果A的个数少于n_A，我们从其他cluster移动一些B类型的点到该cluster中
                cluster_indices = np.where(labels == i)[0]
                other_clusters = np.arange(n_clusters)[n_cluster_after > n_cluster_wished]

                # 在其他cluster中随机选择一些B类型的点，移动到当前cluster中
                for j in other_clusters:
                    B_indices = np.where(labels == j)[0]
                    # currently only for 2 clusters, more than 2 than we need a proportion to give points 
                    # for each cluster with more points
                    indices_to_move = np.random.choice(B_indices, size=n_cluster_after[j] - n_cluster_wished[j], replace=False)

                    # 更新聚类标签和聚类中心
                    labels[indices_to_move] = i
                    centers[i] = np.mean(similarity_matrix[cluster_indices], axis=0)

                    # 更新每个cluster的A类型点个数
                    n_cluster_after = np.array([np.sum(labels == k) for k in range(n_clusters)])
                    if np.all(n_cluster_after >= n_cluster_wished):
                        break

        # 如果没有点被删除或移动，说明已经满足要求，停止迭代
        if np.sum(labels == -1) == 0 and np.all(n_cluster_after >= n_cluster_wished):
            break
        
    # 删除被标记为-1的点
    # valid_indices = np.where(labels != -1)[0]
    # labels = labels[valid_indices]

    # Reassign invalid labels to 0 and 1 (temporary solution)
    invalid_indices = np.where(labels == -1)[0]
    indices_to_be_A = np.random.choice(invalid_indices, size=int(len(invalid_indices)/2), replace=False)
    indices_to_be_B = np.random.choice(invalid_indices, size=len(invalid_indices)-len(indices_to_be_A), replace=False)
              
    labels[indices_to_be_A] = 0
    labels[indices_to_be_B] = 1

    return labels

# # 设置聚类参数
# n_clusters = 2  # 假设我们期望聚类为3类

# # 指定各个cluster中的点的个数
# n_A = 500
# n_B = 500

# # 使用自定义KMeans聚类算法聚类
# labels = custom_kmeans_1(distance_matrix, n_clusters, n_A, n_B)

# # 打印调整后的聚类结果
# print("聚类结果:")
# print(np.count_nonzero(labels == 0))
# print(np.count_nonzero(labels == 1))

# Calculate percentage
def create_type_matrix(type_list):
    n = len(type_list)
    type_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if type_list[i] == type_list[j]:
                type_matrix[i, j] = 1
            else:
                type_matrix[i, j] = 0

    return type_matrix

def calculate_bin_percentage(distance_matrix, type_list, bin_list):
    type_matrix = create_type_matrix(type_list)
    percentage_list = []
    for i, j in zip(bin_list, bin_list[1:]):
        mask_dis = (distance_matrix < j) & (distance_matrix > i)
        mask_dis_type = mask_dis & (type_matrix == 1)
        # distance_matrix_bin  = distance_matrix[(distance_matrix < j) & (distance_matrix > i)]
        count_pairs_all = np.count_nonzero(distance_matrix[mask_dis])
        count_pairs_same = np.count_nonzero(distance_matrix[mask_dis_type])
        percentage_list.append(count_pairs_same / count_pairs_all)

    return percentage_list

# percentage_list = calculate_bin_percentage(distance_matrix, type_list, bin_list)

# # 重新得到聚类后的type_list(谱聚类)
# type_list_clustered = ['A' if label == 0 else 'B' for label in labels]
# percentage_list_clustered = calculate_bin_percentage(distance_matrix, np.array(type_list_clustered), bin_list)

# Visulize synapses on the neuron

import numpy as np
from scipy.optimize import curve_fit

# 定义指数函数模型
def exponential_func(x, a, lambd, b):
    return a * np.exp(-x / lambd) + b

def calculate_error(bin_list, percentage_list, initial_guess=[1, 1, 1]):
    initial_guess = [1, 1, 1]  
    x = np.array(bin_list[1:])
    y_noisy = np.array(percentage_list)
    params, covariance = curve_fit(exponential_func, x, y_noisy, p0=initial_guess, maxfev=10000)

    # 获取拟合结果
    a_fit, lambd_fit, b_fit = params

    # 计算拟合值
    y_fit = exponential_func(x, a_fit, lambd_fit, b_fit)

    # 计算拟合误差
    error = np.sqrt(np.mean((y_noisy - y_fit) ** 2))

    return error,[a_fit, lambd_fit, b_fit]

import random

def partial_shuffle(input_list, percentage=10):
    '''Shuffles any n number of values in a list'''
    count = int(len(input_list)*percentage/100)
    indices_to_shuffle = random.sample(range(len(input_list)), k=count)
    to_shuffle = [input_list[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)
    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        input_list[old_index] = value
    return input_list

numEpoch = 50
shuffle_scale = 10
type_list_fin = type_list

# error_original, initial_guess_original = calculate_error(bin_list, percentage_list)

# for plotting learning rate
error_min_list = []
# while True:
# for i in tqdm(range(numEpoch)):
#     error_list = [error_original]
#     initial_guess_list = [initial_guess_original]
#     type_list_list = [type_list_fin]
#     for j in range(shuffle_scale):
#         type_list_shuffle = partial_shuffle(type_list, percentage=10)
#         percentage_list = calculate_bin_percentage(distance_matrix, type_list_shuffle, bin_list)
#         error, initial_guess = calculate_error(bin_list, percentage_list, initial_guess_original)
#         error_list.append(error)
#         initial_guess_list.append(initial_guess)
#         type_list_list.append(type_list_shuffle)
#     error_original = min(error_list)
#     type_list_fin = type_list_list[error_list.index(error_original)]
#     initial_guess_original = initial_guess_list[error_list.index(error_original)]

#     error_min_list.append(error_original)
#     time.sleep(0.01)
#     # delta_error = error_min - error 
#     # if error < error_min:
#     #     error_min = error 
#     #     type_list_fin = type_list_shuffle 
#     #     initial_guess_min = initial_guess
#     #     print('error: {:.4f}'.format(error))
#     # if error < 0.001:
#     #     break

# type_list_clustered = type_list_fin
# percentage_list_clustered = calculate_bin_percentage(distance_matrix, type_list_clustered, bin_list)

# epochs = list(range(1, numEpoch + 1))
# plt.figure(figsize=(6, 6))
# plt.plot(epochs, error_min_list, marker='o', linestyle='-', color='b', label='Error')
# plt.xlabel('Epoch')
# plt.ylabel('Error')
# # plt.xticks(epochs)
# # plt.grid(True)
# plt.legend()
# # plt.show()

# plt.figure(figsize=(8, 4))

# plt.subplot(1, 2, 1)
# plt.plot(bin_list[1:], percentage_list)
# plt.xlabel('Distance (microns)')
# plt.ylabel('Percentage')
# plt.title('Percentage before clustered')

# plt.subplot(1, 2, 2)
# plt.plot(bin_list[1:], percentage_list_clustered)
# plt.xlabel('Distance (microns)')
# plt.ylabel('Percentage')
# plt.title('Percentage after clustered')

# plt.show()
# distance_matrix_1 = distance_matrix[(type_matrix == 1)
                                    # & (distance_matrix < 2.7) 
                                    # & (distance_matrix > 0)]

# count_matrix = np.count_nonzero(distance_matrix, axis=1)
# count_matrix_1 = np.count_nonzero(distance_matrix_1, axis=1)

# distance_a_a_list = []
# distance_b_b_list = []
# for i in range(len(type_list)):
#     for j in range(len(type_list)):
#         if i < j:
#             if type_list[i] == type_list[j] == 'A' :#and distance_matrix[i,j] < distance_limit:
#                 distance_a_a_list.append(distance_matrix[i,j])
#             if type_list[i] == type_list[j] == 'B' :#and distance_matrix[i,j] < distance_limit:
#                 distance_b_b_list.append(distance_matrix[i,j])

# distance_a_a_clustered_list = []
# distance_b_b_clustered_list = []
# for i in range(len(type_list_clustered)):
#     for j in range(len(type_list_clustered)):
#         if i < j:
#             if type_list_clustered[i] == type_list_clustered[j] == 'A' :#and distance_matrix[i,j] < distance_limit:
#                 distance_a_a_clustered_list.append(distance_matrix[i,j])
#             if type_list_clustered[i] == type_list[j] == 'B' :#and distance_matrix[i,j] < distance_limit:
#                 distance_b_b_clustered_list.append(distance_matrix[i,j])

# distance_list = np.array(distance_list)
# distance_a_a_list = np.array(distance_a_a_list)
# distance_b_b_list = np.array(distance_b_b_list)

# distance_list_1 = distance_list[distance_list < 2.7]
# distance_a_a_list_1 = distance_a_a_list[distance_a_a_list < 2.7]
# distance_b_b_list_1 = distance_b_b_list[distance_b_b_list < 2.7]

## Training
# scale = 100
# for i in range(scale):
#     shuffle(type_list) #10% of [A B B A]

# for i in range(len(ref_list)):
#     if ref_list[i] == tar_list[i] == 'A':
#         distance_a_a_list.append(distance_list[i])
#     if ref_list[i] == tar_list[i] == 'B':
#         distance_b_b_list.append(distance_list[i])


## Visualization
# from networkx.drawing.nx_agraph import graphviz_layout
# pos=graphviz_layout(G) 
# plt.figure()
# nx.draw(G, pos, with_labels=True,
#         node_color=color_map,
#         font_weight='bold',
#         node_size=100)
# plt.show()

# import seaborn as sns
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# sns.histplot(distance_a_a_list, kde=True,stat="density",color='lightskyblue',linewidth=0,label='A-A')
# sns.histplot(distance_b_b_list,kde=True,stat="density",color='orange',linewidth=0,label='B-B')
# sns.histplot(distance_list,kde=True,stat="density",color='lightgreen',linewidth=0,label='All')
# plt.legend()
# plt.xlabel('Distance (microns)')
# plt.ylabel('Probability')
# plt.xlim(0,distance_limit)
# plt.ylim(0,0.004)
# plt.title('Distance distribution before clustered')

# plt.subplot(1, 2, 2)
# sns.histplot(distance_a_a_clustered_list, kde=True,stat="density",color='lightskyblue',linewidth=0,label='A-A')
# sns.histplot(distance_b_b_clustered_list,kde=True,stat="density",color='orange',linewidth=0,label='B-B')
# sns.histplot(distance_list,kde=True,stat="density",color='lightgreen',linewidth=0,label='All')
# plt.legend()
# plt.xlabel('Distance (microns)')
# plt.ylabel('Probability')
# plt.xlim(0,distance_limit)
# plt.ylim(0,0.004)
# plt.title('Distance distribution aftrer clustered')

# plt.show()
#Simulate the full neuron for 1 seconds
soma_v = h.Vector().record(complex_cell.soma[0](0.5)._ref_v)
dend_v = h.Vector().record(complex_cell.dend[0](0.5)._ref_v)
apic_v = h.Vector().record(complex_cell.apic[0](0.5)._ref_v)
time_v = h.Vector().record(h._ref_t)

h.tstop = 1000
st = time.time()
h.run()
print('complex cell simulation time {:.4f}'.format(time.time()-st))

# Plotting the simulation results

plt.figure()
plt.plot(time_v, soma_v, label='soma')
plt.plot(time_v, dend_v, label='basal')
plt.plot(time_v, apic_v, label='apical')

plt.legend()
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.show()