import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import erfc
from torch import nn

from common import load_config, save_results

torch.set_default_dtype(torch.float64)

ACTS = {"tanh": nn.Tanh, "silu": nn.SiLU, "relu": nn.ReLU}


class PINN(nn.Module):
    def __init__(self, hidden, act, dom):
        super().__init__()
        sizes = [2, *hidden, 1]
        layers = []
        for i in range(len(sizes) - 1):
            lin = nn.Linear(sizes[i], sizes[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            if i < len(sizes) - 2:
                layers.append(ACTS[act]())
        self.net = nn.Sequential(*layers)
        self.dom = dom

    def forward(self, x, t):
        d = self.dom
        inp = torch.stack([2 * x / d["X"] - 1, 2 * (t - d["t0"]) / (d["T"] - d["t0"]) - 1], dim=1)
        return self.net(inp)[:, 0]


def u_fn(model, x, t, hard):
    n = model(x, t)
    if not hard:
        return n
    d = model.dom
    s = x / d["X"]
    tau = (t - d["t0"]) / (d["T"] - d["t0"])
    b0 = math.erfc(d["X"] / (2 * math.sqrt(d["t0"])))
    g = torch.erfc(x / (2 * math.sqrt(d["t0"]))) + s * (torch.erfc(d["X"] / (2 * torch.sqrt(t))) - b0)
    return g + 4 * s * (1 - s) * tau * n


def grad_or_zero(out, inp):
    g = torch.autograd.grad(out, inp, torch.ones_like(out), create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(inp) if g is None else g


def derivatives(model, x, t, hard):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = u_fn(model, x, t, hard)
    u_t = grad_or_zero(u, t)
    u_x = grad_or_zero(u, x)
    u_xx = grad_or_zero(u_x, x)
    return u, u_t, u_x, u_xx


def check_hard_bc(c):
    dom = c["domain"]
    torch.manual_seed(0)
    model = PINN(c["hidden"], "tanh", dom)
    t = torch.linspace(dom["t0"], dom["T"], 7)
    x = torch.linspace(0, dom["X"], 7)
    with torch.no_grad():
        left = (u_fn(model, torch.zeros(7), t, True) - 1).abs().max()
        right = (u_fn(model, torch.full((7,), dom["X"]), t, True)
                 - torch.erfc(dom["X"] / (2 * torch.sqrt(t)))).abs().max()
        init = (u_fn(model, x, torch.full((7,), dom["t0"]), True)
                - torch.erfc(x / (2 * math.sqrt(dom["t0"])))).abs().max()
    print(f"hard-BC check on an untrained network (all should be ~0): "
          f"x=0 {left:.1e}, x=X {right:.1e}, t=t0 {init:.1e}")


def train_run(run, c, seed):
    dom = c["domain"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hard = run["hard"]
    model = PINN(c["hidden"], run["act"], dom)

    n = run["n_colloc"]
    xr = torch.tensor(rng.uniform(0, dom["X"], n))
    tr = torch.tensor(rng.uniform(dom["t0"], dom["T"], n))

    nb = c["n_boundary"]
    tb = rng.uniform(dom["t0"], dom["T"], nb)
    xi = rng.uniform(0, dom["X"], nb)
    xb = torch.tensor(np.concatenate([np.zeros(nb), np.full(nb, dom["X"]), xi]))
    tbb = torch.tensor(np.concatenate([tb, tb, np.full(nb, dom["t0"])]))
    target = torch.tensor(np.concatenate([
        np.ones(nb),
        erfc(dom["X"] / (2 * np.sqrt(tb))),
        erfc(xi / (2 * np.sqrt(dom["t0"]))),
    ]))
    weights = torch.tensor(np.concatenate([
        np.full(2 * nb, c["lam_b"] / nb),
        np.full(nb, c["lam_i"] / nb),
    ]))

    def loss_fn():
        _, u_t, _, u_xx = derivatives(model, xr, tr, hard)
        loss = ((u_t - u_xx) ** 2).mean()
        if not hard:
            loss = loss + (weights * (model(xb, tbb) - target) ** 2).sum()
        return loss

    start = time.perf_counter()
    opt = torch.optim.Adam(model.parameters(), lr=c["lr0"])
    for it in range(1, c["adam_iters"] + 1):
        lr = c["lr0"] * (c["lr1"] / c["lr0"]) ** (it / c["adam_iters"])
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if it % c["log_every"] == 0:
            print(f"    adam {it:5d}  loss {loss.item():.3e}  elapsed {time.perf_counter() - start:.0f} s", flush=True)

    if c["lbfgs_iters"] > 0:
        opt2 = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=c["lbfgs_iters"],
            max_eval=int(c["lbfgs_iters"] * 1.5), history_size=50,
            tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            value = loss_fn()
            value.backward()
            return value

        opt2.step(closure)
    seconds = time.perf_counter() - start

    nt = c["n_test"]
    xt, tt = np.meshgrid(np.linspace(0, dom["X"], nt), np.linspace(dom["t0"], dom["T"], nt))
    xt, tt = xt.ravel(), tt.ravel()
    ue = erfc(xt / (2 * np.sqrt(tt)))
    xt_t, tt_t = torch.tensor(xt), torch.tensor(tt)
    with torch.no_grad():
        up = u_fn(model, xt_t, tt_t, hard).numpy()
    rel = float(np.linalg.norm(up - ue) / np.linalg.norm(ue))

    _, _, _, uxx = derivatives(model, xt_t, tt_t, hard)
    uxx_rms = float(uxx.detach().pow(2).mean().sqrt())
    eta = xt / (2 * np.sqrt(tt))
    exact_uxx_rms = float(np.sqrt(np.mean((eta * np.exp(-eta ** 2) / (np.sqrt(np.pi) * tt)) ** 2)))

    return {"rel_l2": rel, "seconds": seconds, "uxx_rms": uxx_rms,
            "exact_uxx_rms": exact_uxx_rms, "profile_T": up[-nt:].tolist()}


def fmt(values, spec):
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{m:{spec}} +/- {s:{spec}}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--adam", type=int, default=None)
    parser.add_argument("--lbfgs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    c = dict(cfg["b2"])
    runs = c["runs"]
    tag = ""
    if args.quick:
        tag = "_quick"
        c.update({"adam_iters": 300, "lbfgs_iters": 0, "seeds": [0], "log_every": 100})
        runs = [dict(r, n_colloc=1000) for r in runs]
        print("QUICK MODE: tiny settings, results are only a crash test")
    elif args.only or args.seeds or args.adam is not None or args.lbfgs is not None:
        tag = "_trial"
        if args.only:
            runs = [runs[int(i)] for i in args.only.split(",")]
        if args.seeds:
            c["seeds"] = [int(v) for v in args.seeds.split(",")]
        if args.adam is not None:
            c["adam_iters"] = args.adam
        if args.lbfgs is not None:
            c["lbfgs_iters"] = args.lbfgs
        print(f"TRIAL MODE: runs {[r['name'] for r in runs]}, seeds {c['seeds']}, "
              f"adam {c['adam_iters']}, lbfgs {c['lbfgs_iters']} (files are saved with a _trial suffix)")

    check_hard_bc(c)

    results = {}
    for run in runs:
        print(f"\n=== {run['name']} ===")
        per_seed = []
        for seed in c["seeds"]:
            res = train_run(run, c, seed)
            per_seed.append(res)
            print(f"  seed {seed}: rel L2 = {res['rel_l2']:.3e}   time = {res['seconds']:.1f} s   "
                  f"rms u_xx = {res['uxx_rms']:.3e} (exact {res['exact_uxx_rms']:.3e})", flush=True)
        results[run["name"]] = per_seed

    lines = ["| Configuration | Relative L2 error (mean +/- std) | Training time, s (mean +/- std) | Seeds below target |",
             "|---|---|---|---|"]
    summary = {}
    for name, per_seed in results.items():
        errs = [r["rel_l2"] for r in per_seed]
        secs = [r["seconds"] for r in per_seed]
        below = sum(e < c["target"] for e in errs)
        lines.append(f"| {name} | {fmt(errs, '.2e')} | {fmt(secs, '.0f')} | {below}/{len(errs)} |")
        summary[name] = {"rel_l2": errs, "seconds": secs,
                         "uxx_rms": [r["uxx_rms"] for r in per_seed],
                         "exact_uxx_rms": per_seed[0]["exact_uxx_rms"]}
    table = "\n".join(lines)
    print("\n" + table)
    with open(f"results/b2_table{tag}.md", "w") as f:
        f.write(table + "\n")
    save_results(f"b2_results{tag}", summary)

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    names = list(results)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, name in enumerate(names):
        errs = np.array([r["rel_l2"] for r in results[name]])
        ax.scatter(np.full(len(errs), i) + np.linspace(-0.12, 0.12, len(errs)), errs,
                   color=colors[i % len(colors)], s=40, zorder=3)
        ax.hlines(errs.mean(), i - 0.25, i + 0.25, color=colors[i % len(colors)], lw=2.5)
    ax.axhline(c["target"], color="k", ls="--", lw=1)
    ax.text(len(names) - 0.5, c["target"] * 1.15, "target 1e-3", ha="right")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(", ", "\n") for n in names], fontsize=8)
    ax.set_ylabel("relative L2 error vs erfc (dimensionless)")
    ax.set_title("PINN ablation: dots = seeds, bar = mean")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"figures/b2_errors{tag}.png", dpi=150)
    plt.close()

    dom = c["domain"]
    xs = np.linspace(0, dom["X"], c["n_test"])
    plt.figure(figsize=(7, 4.6))
    plt.plot(xs, erfc(xs / (2 * math.sqrt(dom["T"]))), "k--", lw=1.5, label="erfc (analytic)")
    for i, name in enumerate(names):
        plt.plot(xs, results[name][0]["profile_T"], color=colors[i % len(colors)], lw=1.8, label=name)
    plt.xlabel("x / L (dimensionless)")
    plt.ylabel("u = C / Cs at t = T (dimensionless)")
    plt.title("PINN profiles at the final time (first seed)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"figures/b2_profiles{tag}.png", dpi=150)
    plt.close()
    print(f"\nsaved figures/b2_errors{tag}.png, figures/b2_profiles{tag}.png, results/b2_table{tag}.md")


if __name__ == "__main__":
    main()
