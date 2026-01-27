import numpy as np

income = np.genfromtxt(
    r"Data/income.csv",
    dtype=float,
    delimiter=",",
    encoding="utf-8-sig"
)
income = income[~np.isnan(income)]
income_sorted = np.sort(income)
n = len(income_sorted)
total_income = income_sorted.sum()
i = np.arange(1, n + 1)
gini = (2 * np.sum(i * income_sorted)) / (n * total_income) - (n + 1) / n

print("Gini coefficient:", round(gini, 3))