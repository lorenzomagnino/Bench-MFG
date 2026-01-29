"""Utility functions for generating directory paths and names with variants and hyperparameters."""

from conf.config_schema import MFGConfig


def get_algorithm_name_with_variant(cfg: MFGConfig) -> str:
    """Get algorithm name with variant appended if applicable.

    Args:
        cfg: MFG configuration

    Returns:
        Algorithm name with variant (e.g., "FP_fictitious_play", "PI_smooth_policy_iteration")
        or just algorithm name if no variant (e.g., "OMD", "PSO")
    """
    algo_target = cfg.algorithm._target_
    if algo_target == "DampedFP":
        variant = cfg.algorithm.dampedfp.lambda_schedule
        return f"{algo_target}_{variant}"
    elif algo_target == "PI":
        variant = cfg.algorithm.pi.variant
        return f"{algo_target}_{variant}"
    else:
        return algo_target


def get_experiment_name_with_hyperparams(cfg: MFGConfig) -> str:
    """Get experiment name with hyperparameters appended.

    Args:
        cfg: MFG configuration

    Returns:
        Experiment name with hyperparameters appended based on algorithm and variant
    """
    base_name = cfg.experiment.name
    algo_target = cfg.algorithm._target_.lower()
    hyperparams = []

    if algo_target == "dampedfp":
        variant = cfg.algorithm.dampedfp.lambda_schedule
        if variant == "damped" and cfg.algorithm.dampedfp.damped_constant is not None:
            damped = cfg.algorithm.dampedfp.damped_constant
            hyperparams.append(f"damped{damped:.2f}".replace(".", "p"))

    elif algo_target == "omd":
        lr = cfg.algorithm.omd.learning_rate
        temp = cfg.algorithm.omd.temperature
        hyperparams.append(f"lr{lr:.4f}".replace(".", "p"))
        hyperparams.append(f"temp{temp:.2f}".replace(".", "p"))

    elif algo_target == "pso":
        temp = cfg.algorithm.pso.temperature
        w = cfg.algorithm.pso.w
        c1 = cfg.algorithm.pso.c1
        c2 = cfg.algorithm.pso.c2
        hyperparams.append(f"temp{temp:.2f}".replace(".", "p"))
        hyperparams.append(f"w{w:.2f}".replace(".", "p"))
        hyperparams.append(f"c1{c1:.2f}".replace(".", "p"))
        hyperparams.append(f"c2{c2:.2f}".replace(".", "p"))

    elif algo_target == "pi":
        variant = cfg.algorithm.pi.variant
        temp = cfg.algorithm.pi.temperature
        hyperparams.append(f"temp{temp:.2f}".replace(".", "p"))
        if (
            variant in ("smooth_policy_iteration", "boltzmann_policy_iteration")
            and cfg.algorithm.pi.damped_constant is not None
        ):
            damped = cfg.algorithm.pi.damped_constant
            hyperparams.append(f"damped{damped:.2f}".replace(".", "p"))

    if hyperparams:
        return f"{base_name}_{'_'.join(hyperparams)}"
    else:
        return base_name


def get_output_directory(cfg: MFGConfig) -> str:
    """Get the output directory path with algorithm variant, seed, and experiment hyperparameters.

    Args:
        cfg: MFG configuration

    Returns:
        Output directory path: outputs/${environment.name}/${algorithm_variant}/seed_${seed}/${experiment_name_with_hyperparams}
        For Garnet environments: outputs/Garnet_{num_states}_{num_actions}_{branching_factor}/Garnet_{instance}/${algorithm_variant}/...
    """
    algo_name = get_algorithm_name_with_variant(cfg)
    exp_name = get_experiment_name_with_hyperparams(cfg)
    env_name = cfg.environment.name
    seed = cfg.experiment.random_seed

    if env_name == "MFGarnet":
        try:
            garnet_config = cfg.environment.reward.mfgarnet
            if garnet_config is not None:
                instance_num = garnet_config.seed
                num_states = cfg.environment.num_states
                num_actions = cfg.environment.num_actions
                branching_factor = garnet_config.branching_factor

                dynamics_structure = garnet_config.dynamics_structure
                reward_structure = garnet_config.reward_structure
                dynamics_abbr = "add" if dynamics_structure == "additive" else "mult"
                reward_abbr = "add" if reward_structure == "additive" else "mult"

                if instance_num is not None:
                    parent_dir = f"Garnet_{num_states}_{num_actions}_{branching_factor}_{dynamics_abbr}_{reward_abbr}"
                    instance_dir = f"Garnet_{instance_num + 1}"  # 1-indexed
                    return f"outputs/{parent_dir}/{instance_dir}/{algo_name}/seed_{seed}/{exp_name}"
        except (AttributeError, KeyError):
            pass

    return f"outputs/{env_name}/{algo_name}/seed_{seed}/{exp_name}"
