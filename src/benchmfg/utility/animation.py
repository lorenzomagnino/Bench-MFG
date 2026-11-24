"""GIF animations of mean field and policy across solver iterations."""

from collections.abc import Sequence
from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

from benchmfg.conf.visualization.visualization_schema import ColorsConfig

# ponytail: 1D (non-grid) envs only; reshape frames to grid_dim for 2D envs if needed.


def plot_gif(
    frames: Sequence[np.ndarray],
    fn: str | Path,
    kind: str = "mean_field",
    fps: int = 3,
    every: int = 1,
    dpi: int = 72,
    action_labels: Sequence[str] | None = None,
) -> Path:
    """Save a GIF of mean fields or policies over solver iterations.

    Args:
        frames: One array per iteration. Mean field: ``(N_states,)``.
            Policy: ``(N_states, N_actions)``.
        fn: Output ``.gif`` path.
        kind: ``"mean_field"`` or ``"policy"``.
        fps: Frames per second.
        every: Keep one frame every ``every`` iterations (last frame always kept).
        dpi: Resolution; lower values give smaller files.
        action_labels: Optional y-axis labels for policy GIFs.

    Returns:
        The path of the written GIF.
    """
    if kind not in ("mean_field", "policy"):
        raise ValueError(f"kind must be 'mean_field' or 'policy', got {kind!r}")
    if len(frames) == 0:
        raise ValueError("frames is empty")

    indices = list(range(0, len(frames), every))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)
    data = [np.asarray(frames[i]) for i in indices]

    colors = ColorsConfig()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if kind == "mean_field":
        n_states = data[0].shape[0]
        artist = ax.bar(np.arange(n_states), data[0], color=colors.mean_field_1d_bar)
        ax.set_ylim(0, max(float(d.max()) for d in data) * 1.1)
        ax.set_xlabel("States")
        ax.set_ylabel("Mass")

        def update(k):
            for bar, h in zip(artist, data[k], strict=False):
                bar.set_height(h)
            ax.set_title(f"Mean field, iteration {indices[k]}")
            return artist
    else:
        n_states, n_actions = data[0].shape
        artist = ax.imshow(
            data[0].T,
            cmap=colors.policy_cmap,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_yticks(np.arange(n_actions))
        if action_labels is not None:
            ax.set_yticklabels(action_labels)
        ax.set_xlabel("States")
        ax.set_ylabel("Actions")
        fig.colorbar(artist, ax=ax)

        def update(k):
            artist.set_data(data[k].T)
            ax.set_title(f"Policy, iteration {indices[k]}")
            return (artist,)

    update(0)
    fig.tight_layout()
    fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, update, frames=len(data)).save(
        fn, writer=PillowWriter(fps=fps), dpi=dpi
    )
    plt.close(fig)
    return fn
