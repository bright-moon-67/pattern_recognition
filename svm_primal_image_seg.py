import numpy as np
from scipy.optimize import minimize

# 主要实现

def solve_primal(X, y, C):
    n, d = X.shape
    # v 中依次存放 w、b 和每个样本的松弛变量。
    start = np.zeros(d + 1 + n) #需要优化的所有的参数w、b、松弛变量
    start[d + 1:] = 1
    A = np.column_stack((y[:, None] * X, y, np.eye(n)))

    def objective(v):
        return 0.5 * v[:d] @ v[:d] + C * v[d + 1:].sum()

    def gradient(v):
        return np.concatenate((v[:d], [0], np.full(n, C)))

    result = minimize(
        objective, start, jac=gradient, method="SLSQP",
        bounds=[(None, None)] * (d + 1) + [(0, None)] * n,
        constraints={"type": "ineq", "fun": lambda v: A @ v - 1, "jac": lambda v: A},
        options={"ftol": 1e-9, "maxiter": 1000}
    )
    return result.x[:d], float(result.x[d])
