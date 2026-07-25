# MF-Garnet: new results vs published table

`z` = |ours - paper| / paper_std. z<=1 is inside the paper's own
seed spread; z>2 would be a real discrepancy.

## 5x5x5 (A/M)

| Algorithm | Paper | Ours | z | |
|---|---|---|---|---|
| Fixed Point (FP) | 1.654 ± 3.3 | 3.294 ± 4.53 (n=5) | 0.50 | ok |
| Damped FP | 1.45 ± 2.85 | 2.256 ± 4.07 (n=5) | 0.28 | ok |
| Fictitious Play | 0.6235 ± 1.65 | 1.243 ± 2.43 (n=5) | 0.37 | ok |
| Boltzmann PI | 1.057 ± 1.86 | 0.4209 ± 0.369 (n=5) | 0.34 | ok |
| Smooth PI | 2.323 ± 4.87 | 4.626 ± 6.78 (n=5) | 0.47 | ok |
| PI | 1.871 ± 3.59 | 3.125 ± 5.14 (n=5) | 0.35 | ok |
| OMD | 2.311 ± 3.08 | 2.971 ± 4.68 (n=5) | 0.21 | ok |
| MF-PSO | 0.225 ± 0.211 | 0.1178 ± 0.235 (n=5) | 0.51 | ok |

## 25x10x10 (A/A)

| Algorithm | Paper | Ours | z | |
|---|---|---|---|---|
| Fixed Point (FP) | 0.00095 ± 0.0017 | 0.003181 ± 0.00334 (n=5) | 1.31 | borderline |
| Damped FP | 0.000851 ± 0.0017 | 0.001895 ± 0.00261 (n=5) | 0.61 | ok |
| Fictitious Play | 0.0026 ± 0.0036 | 0.001744 ± 0.00189 (n=5) | 0.24 | ok |
| Boltzmann PI | 0.9508 ± 0.288 | 0.8324 ± 0.338 (n=5) | 0.41 | ok |
| Smooth PI | 0.00098 ± 0.0026 | 0.001574 ± 0.00248 (n=5) | 0.23 | ok |
| PI | 0.003 ± 0.0046 | 0.0006682 ± 0.000976 (n=5) | 0.51 | ok |
| OMD | 1.437 ± 0.394 | 1.174 ± 0.371 (n=5) | 0.67 | ok |
| MF-PSO | 3.863 ± 1.9 | 2.184 ± 1.72 (n=5) | 0.88 | ok |

Worst deviation: z = 1.31
