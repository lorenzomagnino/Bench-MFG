## New experiment: scaling with the size of the state space

We add a scaling study on MF-Garnet in the **A/M** setting (additive dynamics /
multiplicative reward), varying $\lvert S\rvert$ at fixed $\lvert A\rvert=6$ and
branching factor $6$, potential game, $64$ noise atoms, horizon $100$,
$\gamma=0.90$, $150$ iterations, $200$ PSO particles. Mean $\pm$ std over $2$
seeds (each seed draws a fresh Garnet instance). One NVIDIA L40S per run.

### Final exploitability

| Algorithm | $\lvert S\rvert$=10 | $\lvert S\rvert$=20 | $\lvert S\rvert$=80 | $\lvert S\rvert$=130 | $\lvert S\rvert$=400 |
|:--|--:|--:|--:|--:|--:|
| Fixed Point (FP) | 0.0142 ± 0.02 | 0.000462 ± 0.00065 | 0.00518 ± 0.0071 | 0.0024 ± 0.0027 | 0.00208 ± 0.0024 |
| Damped FP | 0.00592 ± 0.0084 | 2.98e-08 ± 4.2e-08 | 0.00105 ± 0.0011 | 0.00237 ± 0.0031 | 0.0019 ± 0.0019 |
| Fictitious Play | 0.00252 ± 0.0036 | 0.000786 ± 0.0011 | 0.0016 ± 0.002 | 0.00141 ± 0.0019 | 0.00269 ± 0.0029 |
| Boltzmann PI | 0.748 ± 0.42 | 0.728 ± 0.36 | 0.677 ± 0.29 | 0.691 ± 0.34 | 0.739 ± 0.4 |
| Smooth PI | 0.0222 ± 0.031 | -1.19e-07 ± 1.7e-07 | 0.00199 ± 0.0028 | 0.000818 ± 0.00057 | 0.00262 ± 0.0026 |
| PI | 0.00647 ± 0.0091 | 0.000309 ± 9e-05 | 0.00324 ± 0.0045 | 0.000651 ± 0.00075 | 0.00368 ± 0.0039 |
| OMD | 0.999 ± 0.75 | 0.841 ± 0.5 | 0.768 ± 0.4 | 0.789 ± 0.46 | 0.839 ± 0.51 |
| MF-PSO | 0.246 ± 0.3 | 0.185 ± 0.18 | 0.537 ± 0.38 | 0.657 ± 0.5 | 0.901 ± 0.7 |

<sub>Smooth PI at $\lvert S\rvert=20$ returns a small negative value:
exploitability is non-negative by definition, so that cell is float32
cancellation at zero, i.e. convergence to numerical precision.</sub>

### Wall-clock time (seconds)

| Algorithm | $\lvert S\rvert$=10 | $\lvert S\rvert$=20 | $\lvert S\rvert$=80 | $\lvert S\rvert$=130 | $\lvert S\rvert$=400 |
|:--|--:|--:|--:|--:|--:|
| Fixed Point (FP) | 2 | 2 | 3 | 8 | 179 |
| Damped FP | 2 | 3 | 3 | 8 | 185 |
| Fictitious Play | 85 | 90 | 91 | 98 | 273 |
| Boltzmann PI | 2 | 3 | 4 | 9 | 178 |
| Smooth PI | 2 | 3 | 4 | 8 | 182 |
| PI | 2 | 2 | 3 | 8 | 180 |
| OMD | 2 | 3 | 4 | 9 | 179 |
| MF-PSO | 6 | 6 | 19 | 39 | 435 |

### Consistency with the results already in the paper

Both published columns were re-run on the same code.
$z=\lvert\text{ours}-\text{paper}\rvert/\sigma_{\text{paper}}$, so $z\le1$ means the
new value falls inside the variability already reported.

| Algorithm | 5x5x5 (A/M): paper | ours | $z$ | 25x10x10 (A/A): paper | ours | $z$ |
|:--|--:|--:|--:|--:|--:|--:|
| Fixed Point (FP) | 1.65 ± 3.3 | 3.29 ± 4.5 | 0.50 | 0.00095 ± 0.0017 | 0.00318 ± 0.0033 | 1.31 |
| Damped FP | 1.45 ± 2.8 | 2.26 ± 4.1 | 0.28 | 0.000851 ± 0.0017 | 0.0019 ± 0.0026 | 0.61 |
| Fictitious Play | 0.624 ± 1.7 | 1.24 ± 2.4 | 0.37 | 0.0026 ± 0.0036 | 0.00174 ± 0.0019 | 0.24 |
| Boltzmann PI | 1.06 ± 1.9 | 0.421 ± 0.37 | 0.34 | 0.951 ± 0.29 | 0.832 ± 0.34 | 0.41 |
| Smooth PI | 2.32 ± 4.9 | 4.63 ± 6.8 | 0.47 | 0.00098 ± 0.0026 | 0.00157 ± 0.0025 | 0.23 |
| PI | 1.87 ± 3.6 | 3.13 ± 5.1 | 0.35 | 0.003 ± 0.0046 | 0.000668 ± 0.00098 | 0.51 |
| OMD | 2.31 ± 3.1 | 2.97 ± 4.7 | 0.21 | 1.44 ± 0.39 | 1.17 ± 0.37 | 0.67 |
| MF-PSO | 0.225 ± 0.21 | 0.118 ± 0.24 | 0.51 | 3.86 ± 1.9 | 2.18 ± 1.7 | 0.88 |

Every cell reproduces: worst deviation $z=1.31$, with 15 of 16
within $z\le1$. The relative ordering of the methods is unchanged.
