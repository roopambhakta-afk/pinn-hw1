import time

import matplotlib.pyplot as plt
import numpy as np

from common import exact, l2_error, load_config, save_results
from solvers import crank_nicolson_step, explicit_step

cfg = load_config()
c = cfg["b1"]["convergence"]
X, t0, T = cfg["b1"]["X"], cfg["b1"]["t0"], cfg["b1"]["T"]

dxs = c["dx_list"]
errors = {"Crank-Nicolson": [], "Explicit": []}
seconds = {"Crank-Nicolson": [], "Explicit": []}
for name in errors:
    for dx in dxs:
        x = np.arange(0, X + dx / 2, dx)
        u = exact(x, t0)
        if name == "Crank-Nicolson":
            n = round((T - t0) / (c["cn_dt_over_dx"] * dx))
            step = crank_nicolson_step
        else:
            n = int(np.ceil((T - t0) / (c["explicit_r"] * dx ** 2)))
            step = explicit_step
        r = ((T - t0) / n) / dx ** 2
        start = time.perf_counter()
        for _ in range(n):
            u = step(u, r)
        seconds[name].append(time.perf_counter() - start)
        errors[name].append(l2_error(u - exact(x, T), dx))

results = {"dx": dxs}
for name, e in errors.items():
    e = np.array(e)
    orders = np.log2(e[:-1] / e[1:])
    slope = float(np.polyfit(np.log(dxs), np.log(e), 1)[0])
    results[name] = {
        "L2_errors": e.tolist(),
        "pairwise_orders": orders.tolist(),
        "fitted_order": slope,
        "wall_clock_seconds": seconds[name],
    }
    print(name)
    print("  errors :", ["%.2e" % v for v in e])
    print("  orders :", np.round(orders, 2), " fitted:", round(slope, 2))
    print("  seconds:", ["%.4f" % s for s in seconds[name]])

plt.figure(figsize=(6, 4.5))
plt.loglog(dxs, errors["Crank-Nicolson"], "o-", label="Crank-Nicolson")
plt.loglog(dxs, errors["Explicit"], "s-", label=f"Explicit (r = {c['explicit_r']})")
guide = 0.5 * errors["Crank-Nicolson"][-1] * (np.array(dxs) / dxs[-1]) ** 2
plt.loglog(dxs, guide, "k--", label="slope 2")
plt.xlabel("grid spacing dx (dimensionless)")
plt.ylabel(f"L2 error vs erfc at t = {T:g} (dimensionless)")
plt.legend()
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/b1_convergence.png", dpi=150)
save_results("b1_convergence", results)
print("saved figures/b1_convergence.png")
