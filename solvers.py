import numpy as np


def explicit_step(u, r):
    new = u.copy()
    new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
    return new


def thomas(lower, diag, upper, rhs):
    n = len(diag)
    c = np.zeros(n - 1)
    d = np.zeros(n)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def crank_nicolson_step(u, r):
    n_inside = len(u) - 2
    rhs = u[1:-1] + 0.5 * r * (u[2:] - 2 * u[1:-1] + u[:-2])
    rhs[0] += 0.5 * r * u[0]
    rhs[-1] += 0.5 * r * u[-1]
    lower = np.full(n_inside - 1, -0.5 * r)
    upper = np.full(n_inside - 1, -0.5 * r)
    diag = np.full(n_inside, 1 + r)
    new = u.copy()
    new[1:-1] = thomas(lower, diag, upper, rhs)
    return new


def nonlinear_step(u, s, beta):
    D = 1 + beta * u
    D_half = 0.5 * (D[1:] + D[:-1])
    flux = D_half * (u[1:] - u[:-1])
    new = u.copy()
    new[1:-1] = u[1:-1] + s * (flux[1:] - flux[:-1])
    return new
