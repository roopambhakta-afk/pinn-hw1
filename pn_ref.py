import numpy as np

from solvers import thomas

Q = 1.602176634e-19
KB = 1.380649e-23
EPS0 = 8.8541878128e-14


def constants(p):
    Vt = KB * p["T_K"] / Q
    eps = p["eps_r"] * EPS0
    LDi = np.sqrt(eps * Vt / (Q * p["ni"]))
    return Vt, eps, LDi


def make_grid(p, h_nm):
    n = int(round(2 * p["half_width_nm"] / h_nm))
    if n % 2 == 1:
        n += 1
    return (np.arange(n) - (n - 1) / 2) * h_nm


def scaled_doping(x_nm, p):
    return np.where(x_nm > 0, p["ND"], -p["NA"]) / p["ni"]


def solve_equilibrium(p, h_nm, tol=1e-10, max_iter=200):
    Vt, eps, LDi = constants(p)
    lam2 = (LDi / (p["L_nm"] * 1e-7)) ** 2
    x_nm = make_grid(p, h_nm)
    xt = x_nm / p["L_nm"]
    d = xt[1] - xt[0]
    Nt = scaled_doping(x_nm, p)
    psi = np.arcsinh(Nt / 2)
    n = len(psi)
    iterations = 0
    for it in range(max_iter):
        lap = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / d ** 2
        F = lam2 * lap - (2 * np.sinh(psi[1:-1]) - Nt[1:-1])
        diag = -2 * lam2 / d ** 2 - 2 * np.cosh(psi[1:-1])
        off = np.full(n - 3, lam2 / d ** 2)
        delta = thomas(off, diag, off, -F)
        biggest = np.max(np.abs(delta))
        psi[1:-1] += min(1.0, 2.0 / biggest) * delta
        iterations = it + 1
        if biggest < tol:
            break
    return {"x_nm": x_nm, "psi": psi, "Vt": Vt, "eps": eps, "lam2": lam2,
            "h_nm": h_nm, "iterations": iterations}


def summarize(sol, p):
    Vt = sol["Vt"]
    x_cm = sol["x_nm"] * 1e-7
    psi_V = Vt * sol["psi"]
    field = -(psi_V[1:] - psi_V[:-1]) / (x_cm[1:] - x_cm[:-1])
    V_bi = psi_V[-1] - psi_V[0]
    E_max = float(np.max(np.abs(field)))
    return {"V_bi_V": float(V_bi), "E_max_V_per_cm": E_max,
            "W_nm": float(2 * V_bi / E_max * 1e7)}


def analytic(p):
    Vt, eps, LDi = constants(p)
    NA, ND, ni = p["NA"], p["ND"], p["ni"]
    V_bi = Vt * np.log(NA * ND / ni ** 2)
    W = np.sqrt(2 * eps * V_bi / Q * (1 / NA + 1 / ND))
    x_n = W * NA / (NA + ND)
    return {"V_bi_V": float(V_bi), "W_nm": float(W * 1e7),
            "x_n_nm": float(x_n * 1e7), "x_p_nm": float((W - x_n) * 1e7),
            "E_max_V_per_cm": float(Q * ND * x_n / eps)}
