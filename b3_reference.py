import matplotlib.pyplot as plt
import numpy as np

from common import load_config, save_results
from pn_ref import analytic, solve_equilibrium, summarize

cfg = load_config()
p = cfg["b3"]
r = p["ref"]

exact_dep = analytic(p)
print("Depletion approximation (A3):")
for k, v in exact_dep.items():
    print("  %-16s %.6g" % (k, v))

rows = []
print("\nGrid refinement of the finite-difference reference:")
for h in r["convergence_h_nm"]:
    sol = solve_equilibrium(p, h, r["tol"], r["max_iter"])
    s = summarize(sol, p)
    s["h_nm"] = h
    s["newton_iterations"] = sol["iterations"]
    rows.append(s)
    print("  h = %.3f nm  nodes = %5d  newton = %2d  V_bi = %.6f V  E_max = %.5e V/cm  W = %.4f nm"
          % (h, len(sol["x_nm"]), sol["iterations"], s["V_bi_V"], s["E_max_V_per_cm"], s["W_nm"]))

diffs = [abs(rows[i]["W_nm"] - rows[i + 1]["W_nm"]) for i in range(len(rows) - 1)]
orders = [float(np.log2(diffs[i] / diffs[i + 1])) for i in range(len(diffs) - 1)]
print("\nSuccessive differences in W (nm):", ["%.3e" % d for d in diffs])
print("Observed order from W:", np.round(orders, 2))

sol = solve_equilibrium(p, r["h_nm"], r["tol"], r["max_iter"])
ref = summarize(sol, p)
print("\nReference used from here on (h = %g nm):" % r["h_nm"])
for k, v in ref.items():
    print("  %-16s %.6g" % (k, v))
print("W numerical / W depletion approximation = %.4f" % (ref["W_nm"] / exact_dep["W_nm"]))

x = sol["x_nm"]
psi = sol["psi"]
ni = p["ni"]
n = ni * np.exp(psi)
pp = ni * np.exp(-psi)
psi_V = sol["Vt"] * psi

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].plot(x, psi_V, color="#2a78d6")
axes[0].set_xlabel("position x (nm)")
axes[0].set_ylabel("potential (V)")
axes[0].set_title("Potential")

axes[1].plot(x, n, color="#2a78d6", label="electrons n")
axes[1].plot(x, pp, color="#eb6834", label="holes p")
axes[1].set_xlabel("position x (nm)")
axes[1].set_ylabel("carrier density (cm$^{-3}$)")
axes[1].set_title("Carriers, linear axis")
axes[1].legend()

axes[2].semilogy(x, n, color="#2a78d6", label="electrons n")
axes[2].semilogy(x, pp, color="#eb6834", label="holes p")
axes[2].set_xlabel("position x (nm)")
axes[2].set_ylabel("carrier density (cm$^{-3}$)")
axes[2].set_title("Carriers, log axis")
axes[2].legend()

plt.tight_layout()
plt.savefig("figures/b3_reference.png", dpi=150)

save_results("b3_reference", {
    "depletion_approximation": exact_dep,
    "reference": ref,
    "refinement": rows,
    "observed_order_from_W": orders,
})
print("saved figures/b3_reference.png and results/b3_reference.json")
