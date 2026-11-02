import numpy as np
from scipy.optimize import minimize
def solve_dual(X, y, C):
    n = len(y)
    Q = (y[:, None] * X) @ (y[:, None] * X).T
    # 将对偶问题的最大值改写为负目标函数的最小值。
    result = minimize(
        lambda a: 0.5 * a @ Q @ a - a.sum(), np.zeros(n),
        jac=lambda a: Q @ a - 1, method="SLSQP",
        bounds=[(0, C)] * n,
        constraints={"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y},
        options={"ftol": 1e-9, "maxiter": 1000}
    )
    a = result.x
    w = (a * y) @ X
    free = (a > 1e-6) & (a < C - 1e-6)
    b = (y[free] - X[free] @ w).mean()
    return w, float(b), a
