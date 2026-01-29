"""Utility functions for configuration display."""

from typing import Any

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from rich.tree import Tree


def print_config_table(cfg: DictConfig, style: str = "tree") -> None:
    """Print configuration in a nice hierarchical format.

    Args:
        cfg: OmegaConf configuration object
        style: Display style - "tree" (default) or "table"
    """
    if style == "table":
        print_config_table_compact(cfg)
    else:
        print_config_tree(cfg)


def print_config_tree(cfg: DictConfig) -> None:
    """Print configuration in a nice hierarchical tree format.

    Args:
        cfg: OmegaConf configuration object
    """
    console = Console()
    tree = Tree("📋 Configuration", guide_style="bold bright_blue")

    def format_value(value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "[dim]null[/dim]"
        value_str = str(value)
        if isinstance(value, bool):
            return f"[yellow]{value_str}[/yellow]"
        if isinstance(value, (int, float)):
            return f"[blue]{value_str}[/blue]"
        return f"[green]{value_str}[/green]"

    def add_node(parent: Tree, key: str, value: Any, path: str = "") -> None:
        """Recursively add nodes to the tree."""
        full_path = f"{path}.{key}" if path else key

        if OmegaConf.is_dict(value):
            node = parent.add(f"[bold cyan]{key}[/bold cyan]", style="dim")
            for k, v in value.items():
                add_node(node, k, v, full_path)
        elif OmegaConf.is_list(value):
            if len(value) == 0:
                parent.add(f"[bold]{key}[/bold]: [dim]empty list[/dim]")
            else:
                node = parent.add(f"[bold cyan]{key}[/bold cyan]: [dim]{len(value)} item(s)[/dim]")
                for i, item in enumerate(value):
                    if OmegaConf.is_dict(item) or OmegaConf.is_list(item):
                        add_node(node, f"[{i}]", item, full_path)
                    else:
                        node.add(f"  [{i}]: {format_value(item)}")
        else:
            value_str = str(value)
            if len(value_str) > 60:
                truncated = value_str[:57] + "..."
                if isinstance(value, bool):
                    formatted_value = f"[yellow]{truncated}[/yellow]"
                elif isinstance(value, (int, float)):
                    formatted_value = f"[blue]{truncated}[/blue]"
                else:
                    formatted_value = f"[green]{truncated}[/green]"
            else:
                formatted_value = format_value(value)
            parent.add(f"[bold]{key}[/bold]: {formatted_value}")

    for key, value in cfg.items():
        add_node(tree, key, value)

    console.print(tree)


def print_config_table_compact(cfg: DictConfig) -> None:
    """Print configuration in a compact table format with hierarchical paths.

    Args:
        cfg: OmegaConf configuration object
    """
    console = Console()
    table = Table(title="📋 Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Path", style="cyan", no_wrap=False)
    table.add_column("Value", style="green")

    def format_value(value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "[dim]null[/dim]"
        if isinstance(value, bool):
            return f"[yellow]{value}[/yellow]"
        if isinstance(value, (int, float)):
            return f"[blue]{value}[/blue]"
        return str(value)

    def add_rows(value: Any, path: str = "") -> None:
        """Recursively add rows to the table."""
        if OmegaConf.is_dict(value):
            for k, v in value.items():
                new_path = f"{path}.{k}" if path else k
                add_rows(v, new_path)
        elif OmegaConf.is_list(value):
            if len(value) == 0:
                table.add_row(path, "[dim]empty list[/dim]")
            else:
                for i, item in enumerate(value):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    if OmegaConf.is_dict(item) or OmegaConf.is_list(item):
                        add_rows(item, new_path)
                    else:
                        table.add_row(new_path, format_value(item))
        else:
            value_str = format_value(value)
            if len(value_str) > 80:
                value_str = value_str[:77] + "..."
            table.add_row(path, value_str)

    add_rows(cfg)
    console.print(table)
