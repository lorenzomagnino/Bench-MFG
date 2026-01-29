"""Create the MFG environment based on configuration."""

from typing import Tuple

import numpy as np

from conf.config_schema import EnvironmentConfig, MFGConfig
from envs.mfg_model_class import MFGStationary
from utility.config_utils import (
    create_contraction_game_from_config,
    create_four_rooms_aversion2d_from_config,
    create_kinetic_congestion_from_config,
    create_lasry_lions_chain_from_config,
    create_mf_garnet_from_config,
    create_multiple_equilibria_from_config,
    create_no_interaction_chain_from_config,
    create_rock_paper_scissors_from_config,
    create_sis_epidemic_from_config,
    create_strict_contraction_game_from_config,
    initialize_policy_from_logits,
)


def create_environment(
    cfg: MFGConfig,
) -> Tuple[MFGStationary, np.ndarray]:
    """Create the MFG environment and initialize policy based on configuration.

    Args:
        cfg: MFGConfig containing environment and initialization settings.

    Returns:
        Tuple of (environment, initial_policy)
    """
    env_cfg: EnvironmentConfig = cfg.environment

    if env_cfg.name == "LasryLionsChain":
        environment = create_lasry_lions_chain_from_config(env_cfg)
    elif env_cfg.name == "NoInteractionChain":
        environment = create_no_interaction_chain_from_config(env_cfg)
    elif env_cfg.name == "FourRoomsAversion2D":
        environment = create_four_rooms_aversion2d_from_config(env_cfg)
    elif env_cfg.name == "StrictContractionGame":
        environment = create_strict_contraction_game_from_config(env_cfg)
    elif env_cfg.name == "MultipleEquilibriaGame":
        environment = create_multiple_equilibria_from_config(env_cfg)
    elif env_cfg.name == "MFGarnet":
        environment = create_mf_garnet_from_config(env_cfg)
    elif env_cfg.name == "RockPaperScissors":
        environment = create_rock_paper_scissors_from_config(env_cfg)
    elif env_cfg.name == "SISEpidemic":
        environment = create_sis_epidemic_from_config(env_cfg)
    elif env_cfg.name == "KineticCongestion":
        environment = create_kinetic_congestion_from_config(env_cfg)
    elif env_cfg.name == "ContractionGame":
        environment = create_contraction_game_from_config(env_cfg)
    else:
        raise NotImplementedError(f"Environment {env_cfg.name} not implemented yet")

    initial_policy = initialize_policy_from_logits(
        environment,
        cfg.initialization.initialization_type,
        cfg.initialization.init_policy_temp,
    )

    return environment, initial_policy
