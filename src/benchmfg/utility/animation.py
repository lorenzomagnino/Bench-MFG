"""GIF animations of exploitability, mean field and policy across solver iterations."""

import base64
from collections.abc import Sequence
from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

from benchmfg.conf.visualization.visualization_schema import ColorsConfig

KINDS = ("exploitability", "mean_field", "policy")
# Grid action order used by the 2D envs: up, right, down, left, stay.
GRID_ARROWS = ("↑", "→", "↓", "←", "·")


def plot_gif(
    frames: Sequence,
    fn: str | Path,
    kind: str = "mean_field",
    fps: int = 3,
    every: int = 1,
    dpi: int = 72,
    action_labels: Sequence[str] | None = None,
    grid_dim: Sequence[int] | None = None,
    figsize: tuple[float, float] = (4.5, 3.5),
) -> Path:
    """Save a GIF of exploitability, mean fields or policies over solver iterations.

    Args:
        frames: One entry per iteration. Exploitability: a float.
            Mean field: ``(N_states,)``. Policy: ``(N_states, N_actions)``.
        fn: Output ``.gif`` path.
        kind: ``"exploitability"``, ``"mean_field"`` or ``"policy"``.
        fps: Frames per second.
        every: Keep one frame every ``every`` iterations (last frame always kept).
        dpi: Resolution; lower values give smaller files.
        action_labels: Optional y-axis labels for 1D policy GIFs.
        grid_dim: ``(n_rows, n_cols)`` for grid envs. Mean field becomes a
            heatmap; policy shows the most likely action per cell (arrows),
            colored by its probability.
        figsize: Figure size in inches.

    Returns:
        The path of the written GIF.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if len(frames) == 0:
        raise ValueError("frames is empty")

    indices = list(range(0, len(frames), every))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)
    data = [np.asarray(frames[i]) for i in indices]

    colors = ColorsConfig()
    fig, ax = plt.subplots(figsize=figsize)
    title = {
        "exploitability": "Exploitability",
        "mean_field": "Mean field",
        "policy": "Policy",
    }[kind]

    if kind == "exploitability":
        values = np.asarray(frames, dtype=float)
        (line,) = ax.plot([], [], color="#006D6D", linewidth=2)
        (dot,) = ax.plot([], [], "o", color="#006D6D")
        ax.set_xlim(0, len(values) - 1)
        positive = values[values > 0]
        ax.set_yscale("log")
        ax.set_ylim(positive.min() / 2, positive.max() * 2)
        ax.set_xlabel("Iteration")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        def draw(k):
            i = indices[k]
            line.set_data(np.arange(i + 1), values[: i + 1])
            dot.set_data([i], [values[i]])

    elif kind == "mean_field" and grid_dim is not None:
        image = ax.imshow(
            data[0].reshape(grid_dim),
            cmap=colors.mean_field_2d_cmap,
            origin="lower",
            vmin=0.0,
            vmax=max(float(d.max()) for d in data),
        )
        fig.colorbar(image, ax=ax)

        def draw(k):
            image.set_data(data[k].reshape(grid_dim))

    elif kind == "mean_field":
        bars = ax.bar(
            np.arange(data[0].shape[0]), data[0], color=colors.mean_field_1d_bar
        )
        ax.set_ylim(0, max(float(d.max()) for d in data) * 1.1)
        ax.set_xlabel("States")
        ax.set_ylabel("Mass")

        def draw(k):
            for bar, h in zip(bars, data[k], strict=True):
                bar.set_height(h)

    elif grid_dim is not None:
        n_rows, n_cols = grid_dim
        image = ax.imshow(
            data[0].max(axis=1).reshape(grid_dim),
            cmap=colors.policy_cmap,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        fig.colorbar(image, ax=ax)
        symbols = GRID_ARROWS if data[0].shape[1] == len(GRID_ARROWS) else None
        texts = [
            ax.text(s % n_cols, s // n_cols, "", ha="center", va="center", fontsize=14)
            for s in range(n_rows * n_cols)
        ]

        def draw(k):
            probs = data[k].max(axis=1)
            image.set_data(probs.reshape(grid_dim))
            for text, a, p in zip(texts, data[k].argmax(axis=1), probs, strict=True):
                text.set_text(symbols[a] if symbols else str(a))
                text.set_color("white" if p > 0.5 else "black")

    else:
        image = ax.imshow(
            data[0].T,
            cmap=colors.policy_cmap,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_yticks(np.arange(data[0].shape[1]))
        if action_labels is not None:
            ax.set_yticklabels(action_labels)
        ax.set_xlabel("States")
        ax.set_ylabel("Actions")
        fig.colorbar(image, ax=ax)

        def draw(k):
            image.set_data(data[k].T)

    def update(k):
        draw(k)
        ax.set_title(f"{title}, iteration {indices[k]}")

    update(0)
    fig.tight_layout()
    fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=len(data)).save(
        fn, writer=PillowWriter(fps=fps), dpi=dpi
    )
    plt.close(fig)
    return fn


def show_gifs(*paths: str | Path, width: int = 300):
    """Display GIFs side by side in a notebook (wraps on narrow screens)."""
    from IPython.display import HTML

    images = "".join(
        f'<img src="data:image/gif;base64,'
        f'{base64.b64encode(Path(p).read_bytes()).decode()}" width="{width}">'
        for p in paths
    )
    return HTML(f'<div style="display:flex;flex-wrap:wrap;gap:8px">{images}</div>')
