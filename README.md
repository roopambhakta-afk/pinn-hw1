# Physics-Informed Neural Networks, Homework 1

Dopant diffusion and PN-junction reference solvers.

Status: Part B1 (reference finite-difference solver) is complete. B2 (PINN for diffusion) and B3 (drift-diffusion PN junction) are added to this repository as they are finished.

## Reproduce everything

```
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs every experiment in order. Figures are written to `figures/` and every reported number to `results/` as JSON. All settings (grids, time steps, r values, beta values, seed) live in `config.json`; no number is hard-coded in the scripts. The solvers are deterministic, and the seed in `config.json` is applied to NumPy at the start of every script.

## Files

| File | Purpose |
|---|---|
| `config.json` | every setting used by every experiment |
| `solvers.py` | explicit, Crank-Nicolson (Thomas algorithm) and concentration-dependent-D steps |
| `common.py` | shared helpers: config loading, erfc oracle, L2 norm, plot style |
| `b1_blowup.py` | explicit scheme at r = 0.45, 0.50, 0.51 |
| `b1_convergence.py` | grid-refinement study against the erfc solution |
| `b1_dependent_D.py` | D/D0 = 1 + beta*u compared with erfc |
| `run_all.py` | one command that runs all of the above |

## B1 setup

Scaled problem u_t = u_xx on 0 <= x <= 10, u(0) = 1, u(10) = 0, so r = dt/dx^2 and all quantities are dimensionless. The analytic oracle is u = erfc(x / (2 sqrt(t))).

## B1 results (from `results/`)

Explicit scheme blow-up (dx = 0.05, 1500 steps, step initial condition):

| r | max abs u after 1500 steps |
|---|---|
| 0.45 | 1.0 |
| 0.50 | 1.0 |
| 0.51 | 2.9e+21 (first exceeds 2 at step 202) |

Grid refinement, L2 error at t = 1 (L2 norm = sqrt(dx * sum of squared errors)):

| dx | Crank-Nicolson | Explicit (r about 0.4) |
|---|---|---|
| 0.2 | 5.62e-04 | 1.26e-03 |
| 0.1 | 1.40e-04 | 3.22e-04 |
| 0.05 | 3.49e-05 | 8.03e-05 |
| 0.025 | 8.73e-06 | 2.01e-05 |

Observed order: Crank-Nicolson 2.01, 2.00, 2.00 (fitted 2.00); explicit 1.97, 2.00, 2.00 (fitted 1.99).

Concentration-dependent diffusivity at t = 1:

| beta | depth where u = 0.5 | dose (integral of u) |
|---|---|---|
| erfc (analytic) | 0.954 | 1.128 |
| 0 | 0.954 | 1.129 |
| 3 | 1.846 | 1.915 |
| 10 | 3.061 | 3.033 |

## Choices and assumptions

- The convergence study starts from the exact erfc profile at t0 = 0.1 rather than the discontinuous step at t = 0. With the raw step, Crank-Nicolson (dt proportional to dx) drops to an observed order of about 0.5 because of the corner singularity at x = 0, t = 0; the explicit scheme (r fixed) still shows order 2.
- Crank-Nicolson uses dt = 0.5 dx, so the time and space errors are both second order. The explicit scheme keeps r near 0.4, so dt is proportional to dx^2.
- D(C) = D0 + D-(n/ni) is modelled as D/D0 = 1 + beta*u with beta = (D-/D0)(Cs/ni), assuming n is approximately C (full ionization). The values beta = 3 and 10 are illustrative, not measured data. The nonlinear step is explicit and first order in time, with dt kept small through r_max / (1 + beta).
- Wall-clock times are saved in `results/b1_convergence.json` for later comparison. The Crank-Nicolson times include a pure-Python tridiagonal solver, so they overstate its cost relative to a compiled banded solver.
