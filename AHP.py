import numpy as np

A = np.array([
    [1, 2],
    [1/2, 1]
])

eigenvalues, eigenvectors = np.linalg.eig(A)
max_index = np.argmax(eigenvalues.real)
max_eigenvalue = eigenvalues[max_index].real
weight_vector = eigenvectors[:, max_index].real

weights = weight_vector / weight_vector.sum()

print("判断矩阵 A: ")
print(A)

print("\n权重向量 (Economic Indicators, User Experience Indexes): ")
print(weights)