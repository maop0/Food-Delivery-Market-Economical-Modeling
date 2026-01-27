import numpy as np

income = np.array([3333, 5555, 5000, 8888, 9999], dtype=float)
income_sorted = np.sort(income)
n = len(income_sorted)
total_income = income_sorted.sum()
i = np.arange(1, n + 1)
gini = (2 * np.sum(i * income_sorted)) / (n * total_income) - (n + 1) / n

print("Gini coefficient:", round(gini, 3))