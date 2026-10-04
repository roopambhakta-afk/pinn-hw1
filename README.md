# Physics-Informed Neural Networks, Homework 1

Dopant diffusion and PN-junction reference solvers.

Status: Parts B1 (reference finite-difference solver) and B2 (PINN for diffusion) are complete. For B3 (drift-diffusion PN junction) only the finite-difference reference solver was completed; the PINN experiments were not attempted.

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

## B2: PINN from scratch

Goal: train a network u(x,t) for u_t = u_xx with no PINN library, and reach a relative L2 error below 1e-3 against the exact erfc solution.

How to run (settings are all in config.json, seeds 0, 1, 2):

    python b2_pinn.py

Setup:
- Network: 3 hidden layers of 64 units, float64, Xavier init, inputs scaled to [-1, 1].
- Loss: mean squared PDE residual (u_t - u_xx from autograd) plus boundary and initial-condition penalties (weights 10 and 10, 200 points per group) for the soft-BC runs.
- Training: 3000 Adam steps (learning rate 1e-3 decaying to 1e-4), then 500 L-BFGS steps with strong-Wolfe line search.
- Domain: x in [0, 4], t in [0.1, 1]. Left edge u = 1, right edge and start profile from erfc.
- Error: relative L2 on a test grid (n_test in config.json) against the exact erfc solution.
- Hard BC: u = g + d * N, where g satisfies the boundary and initial conditions exactly and d is zero on those edges. Checked on an untrained network: violation 0 at x = 0, 4.8e-35 at x = X and at t = t0.

Results (mean +/- std over 3 seeds, one lever changed per row relative to the baseline):

| Configuration | Relative L2 error | Training time, s | Seeds below 1e-3 |
|---|---|---|---|
| tanh, 1e4 pts, soft BC (baseline) | 6.52e-04 +/- 9.64e-05 | 800 +/- 36 | 3/3 |
| SiLU, 1e4 pts, soft BC | 1.98e-03 +/- 4.88e-04 | 1427 +/- 24 | 0/3 |
| ReLU, 1e4 pts, soft BC | 5.12e-01 +/- 6.39e-04 | 330 +/- 14 | 0/3 |
| tanh, 1e3 pts, soft BC | 8.01e-04 +/- 2.04e-04 | 77 +/- 0 | 2/3 |
| tanh, 1e4 pts, hard BC | 2.50e-04 +/- 2.26e-05 | 838 +/- 11 | 3/3 |

Figures: figures/b2_errors.png (error per configuration) and figures/b2_profiles.png (profiles against the exact solution).

Findings:
- Hard boundary conditions gave the lowest error (2.6 times below the baseline) with a much smaller spread across seeds, at the same cost. The network no longer has to balance the PDE term against the boundary penalties.
- ReLU fails. A ReLU network is piecewise linear, so its second derivative is zero almost everywhere (measured rms u_xx = 0.000, exact 0.349). The residual reduces to u_t = 0, which cannot describe diffusion. All three seeds end at the same error (0.511 to 0.512), so the failure is systematic.
- ReLU's final loss (about 3e-4) is lower than the baseline's, yet its error is 51 percent. A low loss does not mean a correct solution when the loss cannot see the missing term.
- SiLU was slower to converge under the same step budget (error 2.0e-3, loss still falling at the end). It was not tested with more steps, so this is not a claim that SiLU is worse in general.
- With 1e3 points the error was 8.0e-4 against 6.5e-4 for 1e4 points, about 10 times faster. With 3 seeds the difference is within the spread, so it is not shown to be significant.

Limitations: 3 seeds only. Training settings were not tuned beyond one short trial run. All runs on CPU.

Figure notes:
- b2_errors.png: each dot is one seed, each bar is the mean over 3 seeds, the y axis is logarithmic, and the dashed line is the 1e-3 target.
- b2_profiles.png: first seed, at the final time. The tanh, SiLU and hard-BC curves lie on top of the exact erfc curve (relative errors 2.5e-4 to 2e-3, too small for the plot to show), so only the ReLU curve is visibly different. The ReLU curve is made of straight segments and falls to zero far too fast, as expected for a network with no curvature.

## Environment

Tested with Python 3.14.0 on Windows (Git Bash). Exact package versions are in requirements.txt. Seeds and every setting are in config.json.

Setup from a fresh clone:

    git clone https://github.com/roopambhakta-afk/pinn-hw1
    cd pinn-hw1
    python -m venv .venv
    source .venv/Scripts/activate
    pip install -r requirements.txt

On Linux or macOS use `source .venv/bin/activate` instead of the Scripts line.

Run everything (B1, B2 and the B3 reference):

    python run_all.py

B2 takes about 3 hours on a CPU laptop. Use `python b2_pinn.py --quick` for a 1-minute crash test.

## B3: PN junction reference solver (PINN part not attempted)

Only the finite-difference reference for the equilibrium PN junction was done. The raw PINN, its failure plots, and the one-lever-at-a-time fixes were not attempted.

How to run (settings in config.json under "b3"):

    python b3_reference.py

Method: the scaled Poisson equation lambda^2 psi'' = exp(psi) - exp(-psi) - N from A4 is solved with Newton's method on a uniform grid, with the doping step placed on a cell interface. Boundary values are the charge-neutral bulk potentials. Settings: N_A = 1e17, N_D = 1e18, n_i = 1e10 cm^-3, 300 K, domain +/- 500 nm, L = 250 nm.

Results (grid spacing 0.25 nm):

| Quantity | Finite-difference reference | Depletion approximation (A3) |
|---|---|---|
| V_bi (V) | 0.892896 | 0.892896 |
| Peak field (V/cm) | 1.543e5 | 1.585e5 |
| W (nm) | 115.74 | 112.7 |

W is defined here as 2 V_bi divided by the peak field (the width of the triangle with the same area under the field curve). The numerical W is 2.7 percent larger than the depletion approximation because the real depletion edges are smooth.

Verification: grid spacings 1, 0.5, 0.25 and 0.125 nm give W of 115.7556, 115.7431, 115.7399 and 115.7391 nm. The observed order from successive differences is 1.99 and 2.00, so the solver is second order. V_bi is fixed by the boundary values.

Figure: figures/b3_reference.png (potential, and carriers on linear and log axes). The carrier densities span about 15 powers of ten, which is the difficulty a PINN would face.
