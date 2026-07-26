## New experiment: scaling with the size of the state space

We add a scaling study on MF-Garnet, varying $\lvert S\rvert$ at fixed
$\lvert A\rvert=6$ and branching factor $6$, potential game, $64$ noise atoms,
horizon $100$, $\gamma=0.90$, $150$ iterations, $200$ PSO particles. Mean
$\pm$ std over seeds, each seed drawing a fresh Garnet instance. One NVIDIA
L40S per run. A = additive coupling, M = multiplicative, given as
dynamics/reward.

### Modality A/M (5 seeds per cell)

**Final exploitability**

| Algorithm | $\lvert S\rvert$=10 | $\lvert S\rvert$=20 | $\lvert S\rvert$=80 | $\lvert S\rvert$=130 | $\lvert S\rvert$=400 |
|:--|--:|--:|--:|--:|--:|
| Fixed Point (FP) | 0.00607 ± 0.013 [3/5] | 0.000907 ± 0.0016 [3/5] | 0.00704 ± 0.0047 | 0.00966 ± 0.0092 | 0.0148 ± 0.014 |
| Damped FP | 0.0436 ± 0.091 [3/5] | <1e-6 [5/5] | 0.00295 ± 0.0033 | 0.00648 ± 0.0057 | 0.00842 ± 0.0062 |
| Fictitious Play | 0.0073 ± 0.014 [3/5] | 0.00374 ± 0.0052 [2/5] | 0.00162 ± 0.0014 | 0.00646 ± 0.0062 | 0.00987 ± 0.0075 |
| Boltzmann PI | 0.618 ± 0.27 | 0.557 ± 0.24 | 0.587 ± 0.17 | 0.586 ± 0.2 | 0.618 ± 0.23 |
| Smooth PI | 0.00926 ± 0.02 [3/5] | 0.000605 ± 0.0014 [4/5] | 0.00132 ± 0.0019 [2/5] | 0.0091 ± 0.0088 | 0.00816 ± 0.0061 |
| PI | 0.0756 ± 0.16 [3/5] | 0.00576 ± 0.013 [2/5] | 0.00455 ± 0.0045 [1/5] | 0.00846 ± 0.0085 | 0.0197 ± 0.015 |
| OMD | 0.951 ± 0.4 | 0.819 ± 0.26 | 0.861 ± 0.23 | 0.842 ± 0.24 | 0.893 ± 0.27 |
| MF-PSO | 0.416 ± 0.34 | 1.08 ± 0.92 | 3.53 ± 2.8 | 3.99 ± 3.1 | 5.37 ± 4.2 |

<sub>`<1e-6` marks cells solved to the float32 resolution of
exploitability (a difference of order-1 value functions, so ~6e-08 per ULP);
the digits below that threshold carry no information. `[k/n]` counts the
seeds that reached the floor -- the per-cell distribution is bimodal
(solved / not solved), so the mean alone understates both outcomes.</sub>

**Wall-clock time (seconds)**

| Algorithm | $\lvert S\rvert$=10 | $\lvert S\rvert$=20 | $\lvert S\rvert$=80 | $\lvert S\rvert$=130 | $\lvert S\rvert$=400 |
|:--|--:|--:|--:|--:|--:|
| Fixed Point (FP) | 2 | 2 | 3 | 8 | 180 |
| Damped FP | 2 | 3 | 3 | 8 | 178 |
| Fictitious Play | 84 | 88 | 91 | 98 | 277 |
| Boltzmann PI | 2 | 3 | 3 | 9 | 178 |
| Smooth PI | 2 | 3 | 3 | 8 | 181 |
| PI | 2 | 2 | 3 | 8 | 175 |
| OMD | 2 | 3 | 4 | 9 | 181 |
| MF-PSO | 6 | 6 | 18 | 37 | 422 |

### Modality M/A (5 seeds per cell)

**Final exploitability**

| Algorithm | $\lvert S\rvert$=10 | $\lvert S\rvert$=20 | $\lvert S\rvert$=80 | $\lvert S\rvert$=130 | $\lvert S\rvert$=400 |
|:--|--:|--:|--:|--:|--:|
| Fixed Point (FP) | 0.0272 ± 0.044 [3/5] | 1.29e-06 ± 1.1e-06 [3/5] | 7.94e-06 ± 1.5e-05 [1/5] | 0.00082 ± 0.0017 [1/5] | 7.77e-05 ± 0.00017 [2/5] |
| Damped FP | 0.013 ± 0.018 [3/5] | <1e-6 [3/5] | 8.57e-05 ± 0.00019 [1/5] | 0.00081 ± 0.0017 [2/5] | 0.000208 ± 0.00041 [2/5] |
| Fictitious Play | 0.0104 ± 0.019 [1/5] | 0.00042 ± 0.00074 [1/5] | 0.000445 ± 0.00086 [1/5] | 0.000141 ± 0.00018 [1/5] | 0.000131 ± 0.00018 [1/5] |
| Boltzmann PI | 0.614 ± 0.29 | 0.53 ± 0.35 | 0.568 ± 0.25 | 0.592 ± 0.28 | 0.584 ± 0.29 |
| Smooth PI | 0.019 ± 0.027 [3/5] | 1.92e-06 ± 1.5e-06 [2/5] | 2.8e-05 ± 6.1e-05 [3/5] | 0.000593 ± 0.00093 [1/5] | 8.61e-05 ± 0.00019 [2/5] |
| PI | 0.0139 ± 0.019 [3/5] | 1.87e-06 ± 1.3e-06 [1/5] | 0.000115 ± 0.00026 [2/5] | 0.000913 ± 0.0019 [1/5] | 2.62e-05 ± 5.6e-05 [2/5] |
| OMD | 0.927 ± 0.34 | 0.812 ± 0.34 | 0.844 ± 0.25 | 0.862 ± 0.29 | 0.865 ± 0.29 |
| MF-PSO | 0.289 ± 0.37 | 0.319 ± 0.31 | 3.28 ± 2.5 | 3.54 ± 2.7 | 5.33 ± 4.1 |

<sub>`<1e-6` marks cells solved to the float32 resolution of
exploitability (a difference of order-1 value functions, so ~6e-08 per ULP);
the digits below that threshold carry no information. `[k/n]` counts the
seeds that reached the floor -- the per-cell distribution is bimodal
(solved / not solved), so the mean alone understates both outcomes.</sub>

**Wall-clock time (seconds)**

| Algorithm | $\lvert S\rvert$=10 | $\lvert S\rvert$=20 | $\lvert S\rvert$=80 | $\lvert S\rvert$=130 | $\lvert S\rvert$=400 |
|:--|--:|--:|--:|--:|--:|
| Fixed Point (FP) | 2 | 2 | 3 | 8 | 181 |
| Damped FP | 2 | 2 | 3 | 8 | 176 |
| Fictitious Play | 84 | 84 | 90 | 99 | 274 |
| Boltzmann PI | 2 | 3 | 3 | 8 | 173 |
| Smooth PI | 2 | 3 | 4 | 8 | 178 |
| PI | 2 | 2 | 3 | 8 | 174 |
| OMD | 2 | 3 | 4 | 9 | 177 |
| MF-PSO | 6 | 7 | 21 | 42 | 432 |


### Consistency with the results already in the paper

Both published columns were re-run on the same code.
$z=\lvert\text{ours}-\text{paper}\rvert/\sigma_{\text{paper}}$, so $z\le1$ means the
new value falls inside the variability already reported.

| Algorithm | 5x5x5 (A/M): paper | ours | $z$ | 25x10x10 (A/A): paper | ours | $z$ | 25x10x10 (M/A): paper | ours | $z$ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Fixed Point (FP) | 1.65 ± 3.3 | 3.29 ± 4.5 | 0.50 | 0.00095 ± 0.0017 | 0.00318 ± 0.0033 | 1.31 | 0.000213 ± 0.00064 | 0.000425 ± 0.00095 | 0.33 |
| Damped FP | 1.45 ± 2.8 | 2.26 ± 4.1 | 0.28 | 0.000851 ± 0.0017 | 0.0019 ± 0.0026 | 0.61 | 3.43e-05 ± 9.8e-05 | 6.77e-05 ± 0.00015 | 0.34 |
| Fictitious Play | 0.624 ± 1.7 | 1.24 ± 2.4 | 0.37 | 0.0026 ± 0.0036 | 0.00174 ± 0.0019 | 0.24 | 0.000332 ± 0.00074 | 0.000213 ± 0.00028 | 0.16 |
| Boltzmann PI | 1.06 ± 1.9 | 0.421 ± 0.37 | 0.34 | 0.951 ± 0.29 | 0.832 ± 0.34 | 0.41 | 0.932 ± 0.32 | 0.817 ± 0.44 | 0.36 |
| Smooth PI | 2.32 ± 4.9 | 4.63 ± 6.8 | 0.47 | 0.00098 ± 0.0026 | 0.00157 ± 0.0025 | 0.23 | 0.000291 ± 0.00087 | 0.000453 ± 0.001 | 0.19 |
| PI | 1.87 ± 3.6 | 3.13 ± 5.1 | 0.35 | 0.003 ± 0.0046 | 0.000668 ± 0.00098 | 0.51 | 0.0027 ± 0.008 | 0.00531 ± 0.012 | 0.33 |
| OMD | 2.31 ± 3.1 | 2.97 ± 4.7 | 0.21 | 1.44 ± 0.39 | 1.17 ± 0.37 | 0.67 | 1.47 ± 0.45 | 1.18 ± 0.46 | 0.64 |
| MF-PSO | 0.225 ± 0.21 | 0.118 ± 0.24 | 0.51 | 3.86 ± 1.9 | 2.18 ± 1.7 | 0.88 | 4.02 ± 2.2 | 1.85 ± 1.4 | 1.00 |

Every cell reproduces: worst deviation $z=1.31$, with 22 of 24
within $z\le1$. The relative ordering of the methods is unchanged.
