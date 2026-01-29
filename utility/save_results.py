"""Save experimental results."""

from datetime import datetime
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import OmegaConf

from conf.config_schema import MFGConfig
from utility.path_utils import get_output_directory


def save_results(
    results, cfg: MFGConfig, initial_policy=None, initial_mean_field=None
) -> str:
    """Save experimental results with unique run ID.

    Args:
        results: Tuple of optimal policy, mean field, and exploitabilities.
        cfg: MFGConfig
        initial_policy: Optional initial policy to save.
        initial_mean_field: Optional initial mean field to save.

    Returns:
        run_id: Unique identifier for this run (timestamp-based, with job number for sweeps)

    Saves:
    - config.yaml: Full configuration used for this run
    - exploitabilities.npz: Exploitability values through iterations
    - final_mean_field.npz: Final mean field distribution
    - final_policy.npz: Final optimal policy
    - initial_policy.npz: Initial policy (if provided)
    - initial_mean_field.npz: Initial mean field (if provided)
    - plots/: Directory containing visualization plots (handled by plot_results)
    """
    optimal_policy, mean_field, exploitabilities = results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    try:
        hydra_cfg = HydraConfig.get()
        if hydra_cfg is not None:
            job_num = getattr(hydra_cfg.job, "num", None)
            run_id = f"{timestamp}_job{job_num}" if job_num is not None else timestamp
        else:
            run_id = timestamp
    except Exception:
        run_id = timestamp

    base_dir = Path(get_output_directory(cfg))
    output_dir = base_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    exploitabilities_path = output_dir / "exploitabilities.npz"
    np.savez(exploitabilities_path, exploitabilities=exploitabilities)

    mean_field_path = output_dir / "final_mean_field.npz"
    np.savez(mean_field_path, mean_field=mean_field)

    policy_path = output_dir / "final_policy.npz"
    np.savez(policy_path, policy=optimal_policy)

    if initial_policy is not None:
        initial_policy_path = output_dir / "initial_policy.npz"
        np.savez(initial_policy_path, policy=initial_policy)

    if initial_mean_field is not None:
        initial_mean_field_path = output_dir / "initial_mean_field.npz"
        np.savez(initial_mean_field_path, mean_field=initial_mean_field)

    return run_id
