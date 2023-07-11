# from neuron import h

# soma = h.Section(name='soma')
# dend = h.Section(name='dend')
# dend.connect(soma(0.5))

# soma.L = 10
# dend.L = 100

# length = h.distance(soma(0.5),dend(0.9))
# print(length)

import numpy as np
import matplotlib.pyplot as plt

def distance_distribution_uniform(l, n):
    # 计算距离范围
    distances = np.linspace(0, l, num=n)
    # 初始化概率分布数组
    probabilities = np.zeros_like(distances)

    # 计算概率分布
    for i in range(n):
        for j in range(i+1, n):
            # 计算两点之间的距离
            distance = abs((j - i) * l / (n - 1))
            # 查找对应的距离范围索引
            idx = np.abs(distances - distance).argmin()
            # 增加对应距离范围的概率
            probabilities[idx] += 1

    # 归一化概率分布
    probabilities /= n * (n-1) / 2

    # 绘制概率密度曲线
    plt.plot(distances, probabilities)
    plt.xlabel('Distance')
    plt.ylabel('Probability Density')
    plt.title('Uniform Distance Distribution')
    # plt.show()


def distance_distribution_random(l, n, num_samples=100000):
    # 生成随机点的位置
    points = np.random.uniform(0, l, size=n)
    # 初始化距离数组
    distances = np.zeros(num_samples)

    # 生成样本
    for i in range(num_samples):
        # 随机选择两个点
        indices = np.random.choice(n, size=2, replace=False)
        # 计算两点之间的距离
        distance = abs(points[indices[1]] - points[indices[0]])
        # 存储距离
        distances[i] = distance

    # 绘制概率密度曲线
    plt.figure()
    plt.hist(distances, bins=50, density=True, alpha=0.7)
    plt.xlabel('Distance')
    plt.ylabel('Probability Density')
    plt.title('Random Distance Distribution')
    # plt.show()

# 测试函数
l = 10  # 线的长度
n = 1000   # 点的个数
distance_distribution_uniform(l, n)
distance_distribution_random(l, n)
plt.show()

