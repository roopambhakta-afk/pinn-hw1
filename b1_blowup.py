import matplotlib.pyplot as plt
import numpy as np

from common import load_config, save_results
from solvers import explicit_step

cfg = load_config()
c = cfg["b1"]["blowup"]
X = cfg["b1"]["X"]

dx = c["dx"]
x = np.arange(0, X + dx / 2, dx)
u0 = np.zeros_like(x)
u0[0] = 1.0

history = {}
results = {}
for r in c["r_list"]:
    u = u0.copy()
    values = []
    for n in range(c["steps"]):
        u = explicit_step(u, r)
        values.append(float(abs(u).max()))
    history[r] = values
    results[f"r={r}"] = {"max_abs_u_final": values[-1]}
    print(f"r = {r}:  max|u| after {c['steps']} steps = {values[-1]:.3e}")

u_bad = u0.copy()
u_good = u0.copy()
n_bad = 0
while abs(u_bad).max() < c["threshold"]:
    u_bad = explicit_step(u_bad, 0.51)
    u_good = explicit_step(u_good, 0.5)
    n_bad += 1
results["r=0.51"]["first_step_above_threshold"] = n_bad
print(f"r = 0.51 first exceeds {c['threshold']} at step {n_bad}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
colors = {0.45: "#2a78d6", 0.5: "#1baf7a", 0.51: "#eb6834"}
styles = {0.45: ("--", 1.6), 0.5: ("-", 3.0), 0.51: ("-", 2.0)}
for r in [0.5, 0.45, 0.51]:
    ls, lw = styles[r]
    ax[0].semilogy(range(1, c["steps"] + 1), history[r], ls=ls, lw=lw, color=colors[r], label=f"r = {r}")
ax[0].set_xlabel("time step n")
ax[0].set_ylabel("max |u| over the grid (dimensionless)")
ax[0].set_title("Explicit scheme: max |u| vs step")
ax[0].legend()

ax[1].plot(x, u_bad, "o-", ms=3, color=colors[0.51], label=f"r = 0.51, step {n_bad}")
ax[1].plot(x, u_good, color=colors[0.5], label="r = 0.50, same step")
ax[1].set_xlim(0, 4)
ax[1].set_xlabel("x / L (dimensionless)")
ax[1].set_ylabel("u = C / Cs (dimensionless)")
ax[1].set_title(f"Profile when max |u| first exceeds {c['threshold']:g}")
ax[1].legend()

plt.tight_layout()
plt.savefig("figures/b1_blowup.png", dpi=150)
save_results("b1_blowup", results)
print("saved figures/b1_blowup.png")
