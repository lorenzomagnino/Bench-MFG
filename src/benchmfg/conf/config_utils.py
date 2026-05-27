"""Utility functions for configuration display."""

from typing import Any

from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError, InterpolationResolutionError
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from benchmfg.conf.config_schema import MFGConfig


def _gpu_available_but_cpu_selected(cfg: MFGConfig) -> bool:
    """Return True when a JAX GPU backend is available but config uses CPU."""
    if str(cfg.device).lower() != "cpu":
        return False

    try:
        import jax

        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


def _print_device_warning(console: Console, cfg: MFGConfig) -> None:
    """Warn when GPU is available but the config is set to CPU."""
    if _gpu_available_but_cpu_selected(cfg):
        console.print(
            "[bold yellow]Warning:[/bold yellow] GPU detected, but "
            "[bold]device=cpu[/bold]. Consider changing the default config to "
            "[bold]device=cuda[/bold]."
        )


def print_config_table(cfg: MFGConfig, style: str = "tree") -> None:
    """Print configuration in a nice hierarchical format.

    Args:
        cfg: OmegaConf configuration object
        style: Display style - "tree" (default) or "table"
    """
    if style == "table":
        print_config_table_compact(cfg)
    else:
        print_config_tree(cfg)


def print_config_tree(cfg: MFGConfig) -> None:
    """Print configuration in a nice hierarchical tree format.
    Args:
        cfg: OmegaConf configuration object
    """
    console = Console()
    tree = Tree("📋 Configuration", guide_style="bold bright_blue")

    def resolve_leaf_for_display(select_path: str, raw: Any) -> Any:
        """Resolve a leaf via OmegaConf.select when possible (prettier paths)."""
        if not OmegaConf.is_config(cfg):
            return raw
        try:
            return OmegaConf.select(cfg, select_path)
        except (
            InterpolationResolutionError,
            InterpolationKeyError,
            ValueError,
        ):
            return raw

    def format_value(value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "[dim]null[/dim]"
        value_str = str(value)
        if isinstance(value, bool):
            return f"[yellow]{value_str}[/yellow]"
        if isinstance(value, int | float):
            return f"[blue]{value_str}[/blue]"
        return f"[green]{value_str}[/green]"

    def add_node(parent: Tree, label: str, value: Any, select_path: str) -> None:
        """Recursively add nodes; select_path is used for OmegaConf leaf resolution."""
        if OmegaConf.is_dict(value):
            node = parent.add(f"[bold cyan]{label}[/bold cyan]")
            for k, v in value.items_ex(resolve=False):
                child_path = f"{select_path}.{k}"
                add_node(node, str(k), v, child_path)
        elif OmegaConf.is_list(value):
            if len(value) == 0:
                parent.add(f"[bold]{label}[/bold]: [dim]empty list[/dim]")
            else:
                node = parent.add(
                    f"[bold cyan]{label}[/bold cyan]: [dim]{len(value)} item(s)[/dim]"
                )
                for i in range(len(value)):
                    item = value._get_node(i)
                    idx_label = f"[{i}]"
                    idx_path = f"{select_path}.{i}"
                    if OmegaConf.is_dict(item) or OmegaConf.is_list(item):
                        add_node(node, idx_label, item, idx_path)
                    else:
                        disp = resolve_leaf_for_display(idx_path, item)
                        node.add(f"  [{i}]: {format_value(disp)}")
        else:
            disp = resolve_leaf_for_display(select_path, value)
            value_str = str(disp)
            if len(value_str) > 60:
                truncated = value_str[:57] + "..."
                if isinstance(disp, bool):
                    formatted_value = f"[yellow]{truncated}[/yellow]"
                elif isinstance(disp, int | float):
                    formatted_value = f"[blue]{truncated}[/blue]"
                else:
                    formatted_value = f"[green]{truncated}[/green]"
            else:
                formatted_value = format_value(disp)
            parent.add(f"[bold]{label}[/bold]: {formatted_value}")

    for key, value in cfg.items_ex(resolve=False):
        add_node(tree, str(key), value, str(key))

    console.print(tree)
    _print_device_warning(console, cfg)


def print_config_table_compact(cfg: MFGConfig) -> None:
    """Print configuration in a compact table format with hierarchical paths.

    Args:
        cfg: OmegaConf configuration object
    """
    console = Console()
    table = Table(
        title="📋 Configuration", show_header=True, header_style="bold magenta"
    )
    table.add_column("Path", style="cyan", no_wrap=False)
    table.add_column("Value", style="green")

    def format_value(value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "[dim]null[/dim]"
        if isinstance(value, bool):
            return f"[yellow]{value}[/yellow]"
        if isinstance(value, int | float):
            return f"[blue]{value}[/blue]"
        return str(value)

    def resolve_leaf_for_display(select_path: str, raw: Any) -> Any:
        if not OmegaConf.is_config(cfg):
            return raw
        try:
            return OmegaConf.select(cfg, select_path)
        except (
            InterpolationResolutionError,
            InterpolationKeyError,
            ValueError,
        ):
            return raw

    def add_rows(value: Any, select_path: str = "") -> None:
        """Recursively add rows; select_path matches OmegaConf.select dotted form."""
        if OmegaConf.is_dict(value):
            for k, v in value.items_ex(resolve=False):
                child_path = f"{select_path}.{k}" if select_path else str(k)
                add_rows(v, child_path)
        elif OmegaConf.is_list(value):
            if len(value) == 0:
                table.add_row(select_path, "[dim]empty list[/dim]")
            else:
                for i in range(len(value)):
                    item = value._get_node(i)
                    idx_path = f"{select_path}.{i}"
                    if OmegaConf.is_dict(item) or OmegaConf.is_list(item):
                        add_rows(item, idx_path)
                    else:
                        disp = resolve_leaf_for_display(idx_path, item)
                        table.add_row(idx_path, format_value(disp))
        else:
            disp = resolve_leaf_for_display(select_path, value)
            value_str = format_value(disp)
            if len(value_str) > 80:
                value_str = value_str[:77] + "..."
            table.add_row(select_path, value_str)

    add_rows(cfg)
    console.print(table)
    _print_device_warning(console, cfg)
