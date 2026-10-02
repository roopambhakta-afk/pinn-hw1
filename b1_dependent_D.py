import matplotlib.pyplot as plt
import numpy as np

from common import exact, l2_error, load_config, save_results
from solvers import nonlinear_step

cfg = load_config()
c = cfg["b1"]["dependent_D"]
X, T = cfg["b1"]["X"], cfg["b1"]["T"]

dx = c["dx"]
x = np.arange(0, X + dx / 2, dx)


def depth_at(u, level):
    i = np.argmax(u < level)
    return float(x[i - 1] + (u[i - 1] - level) / (u[i - 1] - u[i]) * dx)


def dose(u):
    return float(np.sum(0.5 * (u[1:] + u[:-1])) * dx)


reference = exact(x, T)
reference[0] = 1.0
results = {"erfc_analytic": {"depth_u_0.5": depth_at(reference, 0.5), "dose": dose(reference)}}
print(f"erfc analytic:  depth where u = 0.5 is {depth_at(reference, 0.5):.3f}")

plt.figure(figsize=(6.5, 4.5))
for beta in c["beta_list"]:
    s = c["r_max"] / (1 + beta)
    n = int(np.ceil(T / (s * dx ** 2)))
    s = T / n / dx ** 2
    u = np.zeros_like(x)
    u[0] = 1.0
    for _ in range(n):
        u = nonlinear_step(u, s, beta)
    results[f"beta={beta}"] = {"depth_u_0.5": depth_at(u, 0.5), "dose": dose(u), "steps": n}
    if beta == 0.0:
        results["beta=0.0"]["L2_vs_erfc"] = l2_error(u - reference, dx)
    print(f"beta = {beta:4.1f}:  depth where u = 0.5  is  {depth_at(u, 0.5):.3f}   dose = {dose(u):.3f}")
    plt.plot(x, u, label=f"beta = {beta:g}")

plt.plot(x, reference, "k--", label="erfc (analytic)")
plt.xlim(0, 6)
plt.xlabel("x / L (dimensionless)")
plt.ylabel("u = C / Cs (dimensionless)")
plt.title(f"D/D0 = 1 + beta u,  t = {T:g}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/b1_dependent_D.png", dpi=150)
save_results("b1_dependent_D", results)
print("saved figures/b1_dependent_D.png")
