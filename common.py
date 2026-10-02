import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from scipy.special import erfc

plt.rcParams.update({
    "axes.prop_cycle": cycler(color=["#2a78d6", "#eb6834", "#1baf7a"]),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "legend.frameon": False,
})


def load_config():
    with open("config.json") as f:
        cfg = json.load(f)
    np.random.seed(cfg["seed"])
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    return cfg


def save_results(name, data):
    with open(f"results/{name}.json", "w") as f:
        json.dump(data, f, indent=2)


def exact(x, t):
    return erfc(x / (2 * np.sqrt(t)))


def l2_error(err, dx):
    return float(np.sqrt(dx * np.sum(err ** 2)))
