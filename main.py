"""Main entry point for the research template."""

import hydra
import wandb

from config.config_schema import Config
from config.config_utils import print_config_table


def init_wandb(cfg: Config) -> None:
    """Initialize WandB."""
    wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        mode=cfg.logging.wandb.mode,
        tags=cfg.logging.wandb.tags,
    )


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: Config) -> None:
    """Main function with Hydra configuration."""

    print_config_table(cfg, style="table")
    # Alternative: use tree format with print_config_table(cfg, style="tree")

    # Optional: init_wandb(cfg)

    # TODO: Add your training/inference logic here


if __name__ == "__main__":
    main()
