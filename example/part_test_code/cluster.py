# # import numpy as np
# # from sklearn.cluster import DBSCAN
# # import matplotlib.pyplot as plt

# # # 生成示例数据
# # np.random.seed(0)
# # n_points = 100
# # points = np.random.rand(n_points, 2)  # 生成随机点坐标，假设是2D坐标
# # types = np.random.choice(['A', 'B'], n_points)  # 随机给每个点分配类型A或B

# # # 将类型转换为数字（0表示A，1表示B）
# # type_num = np.where(types == 'A', 0, 1)

# # # 计算每个点之间的距离（这里使用欧氏距离作为示例）
# # distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

# # # 设置DBSCAN参数，eps表示邻域距离阈值，min_samples表示最小样本数
# # eps = 0.2
# # min_samples = 5

# # # 创建并训练DBSCAN模型
# # dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
# # labels = dbscan.fit_predict(distances)

# # # 可视化聚类结果
# # unique_labels = np.unique(labels)
# # n_clusters = len(unique_labels) - 1  # 去除噪声点（标签为-1）
# # colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

# # for label, color in zip(unique_labels, colors):
# #     if label == -1:
# #         # 跳过噪声点
# #         continue

# #     cluster_points = points[labels == label]
# #     cluster_type = types[labels == label][0]  # 聚类内的点类型都一样
# #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=f'Cluster {label} ({cluster_type})')

# # plt.legend()
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.title('DBSCAN Clustering')
# # plt.show()

# # import numpy as np
# # from sklearn.cluster import SpectralClustering
# # import matplotlib.pyplot as plt

# # # 生成示例数据
# # np.random.seed(0)
# # n_points = 100
# # points = np.random.rand(n_points, 2)  # 生成随机点坐标，假设是2D坐标
# # # 计算每个点之间的距离（这里使用欧氏距离作为示例）
# # distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

# # types = np.random.choice(['A', 'B'], n_points)  # 随机给每个点分配类型A或B

# # # 将类型转换为数字（0表示A，1表示B）
# # type_num = np.where(types == 'A', 0, 1)

# # # 设置谱聚类参数
# # n_clusters = 2  # 假设我们期望聚为2类
# # affinity_matrix = np.exp(-distances ** 2 / (2 * np.median(distances) ** 2))  # 使用高斯核函数作为相似度矩阵

# # # 创建并训练谱聚类模型
# # spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
# # labels = spectral_clustering.fit_predict(affinity_matrix)

# # # 可视化聚类前的散点图
# # plt.figure(figsize=(8, 4))

# # # 聚类前的散点图
# # plt.subplot(1, 2, 1)
# # plt.scatter(points[type_num == 0][:, 0], points[type_num == 0][:, 1], c='b', label='Type A')
# # plt.scatter(points[type_num == 1][:, 0], points[type_num == 1][:, 1], c='r', label='Type B')
# # plt.legend()
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.title('Scatter Plot Before Clustering')

# # # 聚类后的散点图
# # plt.subplot(1, 2, 2)
# # unique_labels = np.unique(labels)
# # colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

# # for label, color in zip(unique_labels, colors):
# #     cluster_points = points[labels == label]
# #     cluster_type = types[labels == label][0]  # 聚类内的点类型都一样
# #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=f'Cluster {label} ({cluster_type})')

# # plt.legend()
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.title('Spectral Clustering')

# # plt.tight_layout()
# # # plt.show()

# # type_list_clustered = np.array(['A' if label == 0 else 'B' for label in labels])

# # # 打印聚类后的type_list
# # print(types)
# # print(np.count_nonzero(types == 'A'))
# # print("聚类后的type_list:")
# # print(type_list_clustered)
# # print(np.count_nonzero(type_list_clustered == 'A'))

# # import numpy as np
# # from sklearn.cluster import KMeans
# # import matplotlib.pyplot as plt

# # # 生成示例数据
# # np.random.seed(0)
# # n_points = 100
# # points = np.random.rand(n_points, 2)  # 生成随机点坐标，假设是2D坐标
# # types = np.random.choice(['A', 'B'], n_points)  # 随机给每个点分配类型A或B

# # # 将类型转换为数字（0表示A，1表示B）
# # type_num = np.where(types == 'A', 0, 1)

# # # 计算每个点之间的距离（这里使用欧氏距离作为示例）
# # distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

# # # 设置KMeans聚类参数
# # n_clusters = 2  # 假设我们期望聚为2类

# # # 创建并训练KMeans聚类模型
# # kmeans = KMeans(n_clusters=n_clusters, random_state=0)
# # labels = kmeans.fit_predict(points)

# # # 重新得到聚类后的type_list
# # type_list_clustered = np.array(['A' if label == 0 else 'B' for label in labels])

# # # 可视化聚类前的散点图
# # plt.figure(figsize=(8, 4))

# # # 聚类前的散点图
# # plt.subplot(1, 2, 1)
# # plt.scatter(points[type_num == 0][:, 0], points[type_num == 0][:, 1], c='b', label='Type A')
# # plt.scatter(points[type_num == 1][:, 0], points[type_num == 1][:, 1], c='r', label='Type B')
# # plt.legend()
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.title('Scatter Plot Before Clustering')

# # # 聚类后的散点图
# # plt.subplot(1, 2, 2)
# # unique_labels = np.unique(labels)
# # colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

# # for label, color in zip(unique_labels, colors):
# #     cluster_points = points[labels == label]
# #     cluster_type = types[labels == label][0]  # 聚类内的点类型都一样
# #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], label=f'Cluster {label} ({cluster_type})')

# # plt.legend()
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.title('Spectral Clustering')

# # plt.tight_layout()
# # plt.show()

# import numpy as np
# from sklearn.cluster import KMeans

# def custom_kmeans(distance_matrix, n_clusters, n_A, n_B, max_iters=1000):
#     # 转换距离矩阵为相似性矩阵（通过取倒数）
#     similarity_matrix = 1 / (1 + distance_matrix)

#     # 初始化聚类中心（可以随机选择n_clusters个点作为初始中心）
#     center_indices = np.random.choice(len(similarity_matrix), size=n_clusters, replace=False)
#     centers = similarity_matrix[center_indices]

#     # 进行KMeans迭代更新
#     for iter_num in range(max_iters):
#         # 计算每个点到聚类中心的距离
#         distances = np.linalg.norm(centers[:, np.newaxis] - similarity_matrix, axis=2)

#         # 确定每个点的聚类标签
#         labels = np.argmin(distances, axis=0) 

#         # 根据指定的n_A和n_B来调整聚类中心
#         for i in range(n_clusters):
#             cluster_indices = np.where(labels == i)[0]
#             n_cluster_A = np.sum(labels[cluster_indices] == 0)
#             n_cluster_B = len(cluster_indices) - n_cluster_A

#             if n_cluster_A > n_A:
#                 # 如果A的个数多于n_A，我们删除一些A类型的点
#                 indices_to_remove = np.random.choice(cluster_indices[labels[cluster_indices] == 0], size=n_cluster_A - n_A, replace=False)
#                 labels[indices_to_remove] = -1  # 将删除的点标记为-1

#             if n_cluster_B > n_B:
#                 # 如果B的个数多于n_B，我们删除一些B类型的点
#                 indices_to_remove = np.random.choice(cluster_indices[labels[cluster_indices] == 1], size=n_cluster_B - n_B, replace=False)
#                 labels[indices_to_remove] = -1  # 将删除的点标记为-1

#             # 更新聚类中心
#             cluster_indices = np.where(labels == i)[0]
#             centers[i] = np.mean(similarity_matrix[cluster_indices], axis=0)

#         # 如果没有点被删除，说明已经满足要求，停止迭代
#         if np.sum(labels == -1) == 0:
#             break

#     print(iter_num)       
#     # 删除被标记为-1的点
#     valid_indices = np.where(labels != -1)[0]
#     labels = labels[valid_indices]

#     return labels

# # 示例使用
# n_points = 100
# distance_matrix = np.random.rand(n_points, n_points)  # 这里使用随机生成的距离矩阵，你应该替换为你的实际距离矩阵

# # 设置聚类参数
# n_clusters = 2  # 假设我们期望聚类为3类

# # 指定各个cluster中的点的个数
# n_A = 40  
# n_B = 60  

# # 使用自定义KMeans聚类算法聚类
# labels = custom_kmeans(distance_matrix, n_clusters, n_A, n_B)

# # 打印调整后的聚类结果
# print("聚类结果:")
# print(labels)
# print(np.count_nonzero(labels == 0))
# print(np.count_nonzero(labels == 1))

import numpy as np

n_clusters = 3
n_cluster_after = np.array([10,20,30])
n_cluster_wished = np.array([10,10,10])
other_clusters = np.arange(n_clusters)[n_cluster_after > n_cluster_wished]
print(other_clusters)