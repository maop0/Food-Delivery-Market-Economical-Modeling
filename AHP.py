import numpy as np

# 构造判断矩阵
# 经济指标比用户体验指标“稍微重要”
A = np.array([
    [1, 2],
    [1/2, 1]
])

# 计算最大特征值和对应特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
max_index = np.argmax(eigenvalues.real)
max_eigenvalue = eigenvalues[max_index].real
weight_vector = eigenvectors[:, max_index].real

# 权重归一化
weights = weight_vector / weight_vector.sum()

print("判断矩阵 A：")
print(A)


print("\n权重向量（Economic, User Experience）：")
print(weights)