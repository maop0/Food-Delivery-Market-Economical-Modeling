import numpy as np
import runpy

# 1. 参数空间（algorithm parameters）
delivery_time = [20, 25, 30, 35, 40, 45, 50]  # 单笔配送时间限制（分钟）
delivery_fee = [0, 1, 2, 3, 4, 5, 6]          # 配送费（元）
order_number = [1, 2, 3, 4, 5]                # 每次接单数
delivery_distance = [1, 2, 3, 4, 5]           # 配送距离（km）
working_time_limit = [8, 10, 12]              # 日工作上限（小时）
minimum_wage = [5000, 6500, 8000]             # 最低工资（元/月）
tax_rate = [0, 0.01, 0.03, 0.05]              # 税率

# 生成所有方案（一个方案 = 一组参数）
schemes = []
for t in delivery_time:
    for f in delivery_fee:
        for o in order_number:
            for d in delivery_distance:
                for w in working_time_limit:
                    for m in minimum_wage:
                        for tax in tax_rate:
                            schemes.append([t, f, o, d, w, m, tax])

schemes = np.array(schemes)

_gini_module = runpy.run_path("Gini Coefficient.py")
BASE_GINI = _gini_module["gini_coefficient"]("Data/income.csv")

# 2. 指标计算函数（你只需要改这里）
def economic_indicators(params):
    """
    Economic indicators:
    [Gini, Externality, Market Stability, Total Surplus, Market Barrier, Unemployment]
    """
    t, f, o, d, w, m, tax = params

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # 这里全部是【示意公式】，你可以随意替换
    gini = BASE_GINI
    externality = 5 + 0.5 * d
    market_stability = 100 - 2 * t
    total_surplus = 200 + 15 * o - 5 * tax
    market_barrier = 0.25
    unemployment = max(0, 10 - o)

    return [
        gini,              # 成本型
        externality,       # 成本型
        market_stability,  # 效益型
        total_surplus,     # 效益型
        market_barrier,    # 成本型
        unemployment       # 成本型
    ]


def user_experience_indicators(params):
    """
    User experience indicators:
    [Waiting Time, Delivery Fee, Courier Workload, Courier Income]
    """
    t, f, o, d, w, m, tax = params

    waiting_time = d * 8 + t * 0.3
    courier_workload = o * 1.5
    courier_income = m * w + o * 200 - tax * 100

    return [
        waiting_time,      # 成本型
        f,                 # 成本型
        courier_workload,  # 成本型
        courier_income     # 效益型
    ]

# 3. 熵权法函数
def entropy_weight(matrix, indicator_type):
    """
    matrix: m x n (方案 × 指标)
    indicator_type: 1 = 效益型, -1 = 成本型
    """
    X = np.array(matrix, dtype=float)
    m, n = X.shape

    # 归一化
    X_norm = np.zeros_like(X)
    for j in range(n):
        if indicator_type[j] == 1:
            X_norm[:, j] = (X[:, j] - X[:, j].min()) / (X[:, j].max() - X[:, j].min())
        else:
            X_norm[:, j] = (X[:, j].max() - X[:, j]) / (X[:, j].max() - X[:, j].min())

    X_norm += 1e-12  # 防止 log(0)

    # 计算熵
    P = X_norm / X_norm.sum(axis=0)
    E = - (P * np.log(P)).sum(axis=0) / np.log(m)

    # 权重
    d = 1 - E
    w = d / d.sum()

    # 子分数
    score = X_norm @ w

    return w, score

# 4. 计算所有方案的指标矩阵
economic_matrix = []
user_matrix = []

for s in schemes:
    economic_matrix.append(economic_indicators(s))
    user_matrix.append(user_experience_indicators(s))

# 指标类型定义
economic_type = [-1, -1, 1, 1, -1, -1]
user_type = [-1, -1, -1, 1]

# 5. 熵权法计算子分数

eco_weights, eco_scores = entropy_weight(economic_matrix, economic_type)
user_weights, user_scores = entropy_weight(user_matrix, user_type)

# 6. AHP 加权合成总分

final_score = 0.667 * eco_scores + 0.333 * user_scores

best_index = np.argmax(final_score)
best_scheme = schemes[best_index]

# 7. 输出结果

print("Best scheme (t, f, o, d, w, m, tax):")
print(best_scheme)

print("Final score:", final_score[best_index])