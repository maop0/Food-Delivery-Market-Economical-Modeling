import numpy as np

def gini_coefficient(source) -> float:
    if isinstance(source, (str, bytes, bytearray, np.str_)):
        values = np.genfromtxt(
            source,
            dtype=float,
            delimiter=",",
            encoding="utf-8-sig"
        )
    else:
        values = np.array(source, dtype=float)

    values = values[~np.isnan(values)]
    values = np.sort(values)
    n = len(values)
    total = values.sum()
    if n == 0 or total == 0:
        return np.nan
    i = np.arange(1, n + 1)
    return (2 * np.sum(i * values)) / (n * total) - (n + 1) / n

if __name__ == "__main__":
    gini = gini_coefficient(r"Data/income.csv")
    print("Gini coefficient:", round(gini, 3))
