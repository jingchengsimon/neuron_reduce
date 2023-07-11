import numpy as np

# 假设有一个 5x5 的二维数组
arr = np.random.randint(1, 10, (5, 5))
print("原始数组：")
print(arr)

# 一个与原始数组相同大小的逻辑条件数组
mask = arr % 2 == 0
mask = mask & (arr > 5)
print("\n逻辑条件数组：")
print(mask)

# 使用逻辑索引，并保持结果数组大小和原始数组一样
result = arr[np.ix_(mask.any(1), mask.any(0))]
print("\n索引后的数组：")
print(result)
