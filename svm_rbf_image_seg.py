import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def rbf(X, Z, gamma):
    return np.exp(-gamma * cdist(X, Z, "sqeuclidean"))


def solve_rbf(X, y, C=1.0, gamma=0.1):
    K = rbf(X, X, gamma)
    Q = y[:, None] * y[None, :] * K
    result = minimize(
        lambda a: 0.5 * a @ Q @ a - a.sum(), np.zeros(len(y)),
        jac=lambda a: Q @ a - 1, method="SLSQP",
        bounds=[(0, C)] * len(y),
        constraints={"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y},
        options={"ftol": 1e-9, "maxiter": 1000}
    )
    a = result.x
    free = (a > 1e-6) & (a < C - 1e-6)
    b = (y[free] - K[free] @ (a * y)).mean()
    return a, b


def predict_rbf(features, X, y, a, b, gamma):
    result = np.empty(len(features), dtype=np.uint8)
    # 分批计算核矩阵，减少整图预测的内存占用。
    for start in range(0, len(features), 4096):
        score = rbf(features[start:start + 4096], X, gamma) @ (a * y) + b
        result[start:start + 4096] = (score >= 0) * 255
    return result

