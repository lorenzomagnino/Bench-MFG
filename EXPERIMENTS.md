# Experiments

```bash
#select algorithm/damped_fixed_point in defaults.yaml
./scripts/run_dampedfp.sh

#select algorithm/omd in defaults.yaml
./scripts/run_omd.sh

#select algorithm/pi in defaults.yaml
./scritps/run_pi.sh

#select algorithm/pso in defaults.yaml
./scripts/run_pso.sh
```

After that we can go to utils/plot_npz/results.py **modified line 1785** with the name of the env that we are running, and corresponding plot_mean_field and plot_policy to accept in case 2d envs. Then run :
```bash
python utility/plot_npz_results.py
```
