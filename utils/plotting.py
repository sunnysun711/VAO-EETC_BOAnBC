# utils/plotting.py
"""Plotting utilities for VAO-EETC-BD project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from shutil import which

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import gurobipy as gp


def latexify() -> bool:
    """Configure matplotlib for LaTeX rendering if available.

    Returns:
        True if LaTeX was found and configured, False otherwise
    """
    if which('latex'):
        params = {
            "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.usetex": True,
            "font.family": "serif",
        }
        matplotlib.rcParams.update(params)
        return True
    return False


def _unwrap(varname: str, variables: dict[str, gp.tupledict] | dict[str, dict[int, float]]) -> dict[int, float]:
    """Unwrap Gurobi variable values.

    Args:
        varname: Variable name
        variables: Optimization variables (gp.tupledict or dict[int, float])

    Returns:
        Dictionary of variable values
    """
    return {
        k: v if isinstance(v, (float, int)) else v.X  # type: ignore
        for k, v in variables[varname].items()
    }


def draw_vao(
    ax: Axes,
    cfg: Any, 
    variables: dict[str, gp.tupledict] | dict[str, dict[int, float]],
    *,
    show_ground: bool = True,
    show_track: bool = True,
    show_tunnel: bool = True,
    show_bridge: bool = True,
    show_vpi: bool = True,
    style_ground: dict | None = None,
    style_track: dict | None = None,
    style_tunnel: dict | None = None,
    style_bridge: dict | None = None,
    style_vpi: dict | None = None,
) -> None:
    """Draw vertical alignment optimization (VAO) results.

    Args:
        ax: Target matplotlib axes
        cfg: Ground configuration (GroundConfigBase)
        variables: Optimization variables (gp.tupledict or dict[int, float])
        show_*: Switches for different elements
        style_*: Matplotlib style overrides
        Default styles:
        style_ground = dict(ls="-", lw=0.7, alpha=0.8, label="ground")
        style_track = dict(ls="-", lw=0.7, alpha=0.8, label="track")
        style_tunnel = dict(color="red", lw=4, alpha=0.3, label="tunnel")
        style_bridge = dict(color="green", lw=4, alpha=0.3, label="bridge")
        style_vpi = dict(marker="x", color="gray", label="VPI")
    """
    S, ds, e6g = cfg.S, cfg.ds, cfg.e6g
    latexify()

    def _draw_segments(indices, yvals, *, color, label, lw, alpha):
        if not indices:
            return

        segments, seg = [], [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                seg.append(indices[i])
            else:
                segments.append(seg)
                seg = [indices[i]]
        segments.append(seg)

        for j, s in enumerate(segments):
            ax.plot(
                s,
                [yvals[i - 1] for i in s],
                color=color,
                lw=lw,
                alpha=alpha,
                label=label if j == len(segments) - 1 else None,
            )

    # Unwrap variables
    e = list(_unwrap("e", variables).values())
    pi = _unwrap("pi", variables)
    z1 = _unwrap("z1", variables) if "z1" in variables else None
    z3 = _unwrap("z3", variables) if "z3" in variables else None

    x = range(1, S + 2)

    # Default styles
    style_ground = style_ground or dict(ls="-", lw=0.7, alpha=0.8, label="ground")
    style_track = style_track or dict(ls="-", lw=0.7, alpha=0.8, label="track")
    style_tunnel = style_tunnel or dict(color="red", lw=4, alpha=0.3, label="tunnel")
    style_bridge = style_bridge or dict(color="green", lw=4, alpha=0.3, label="bridge")
    style_vpi = style_vpi or dict(marker="x", color="gray", label="VPI")

    # Draw ground
    if show_ground:
        if "label" not in style_ground:
            style_ground["label"] = "ground"
        ax.plot(x, e6g, **style_ground)

    # Draw track
    if show_track:
        if "label" not in style_track:
            style_track["label"] = "track"
        ax.plot(x, e, **style_track)

    # VPI points
    if show_vpi:
        vpi_x = [i for i in x if pi[i] > 0.5]
        ax.scatter(
            vpi_x,
            [e[i - 1] for i in vpi_x],
            **style_vpi,
        )

    # Tunnel segments
    if show_tunnel and z1 is not None:
        tunnel_x = [i for i in x if z1[i] > 0.5]
        if "label" not in style_tunnel:
            style_tunnel["label"] = "tunnel"
        _draw_segments(tunnel_x, e, **style_tunnel)

    # Bridge segments
    if show_bridge and z3 is not None:
        bridge_x = [i for i in x if z3[i] > 0.5]
        if "label" not in style_bridge:
            style_bridge["label"] = "bridge"
        _draw_segments(bridge_x, e, **style_bridge)

    # Axes labels
    ax.set_xlabel(f"Horizontal Intervals ({ds} m)")
    ax.set_ylabel("Elevation (m)")
    ax.legend(loc="best", ncols=2)


def draw_tc(
    axes: list[Axes], 
    variables: dict[str, gp.tupledict] | dict[str, dict[Any, float]], 
    ds: float = 50,
    label_suffix: str = "",
    ls: str = "-",
    plot_time: bool = True,
) -> None:
    """Draw train control optimization results.

    Args:
        axes: List of 3 matplotlib axes [velocity, forces, resistances]
        variables: Optimization variables
        ds: Segment length
    """
    latexify()
    
    S = len(_unwrap("v", variables)) - 1
    v = _unwrap("v", variables)                                     # 1 -> S+1
    uPlus = _unwrap("uPlus", variables)                             # 1 -> S
    uMinus = _unwrap("uMinus", variables)                           # 1 -> S
    alpha = _unwrap("alpha", variables)                             # 1 -> S+1
    t = {k: ds * alpha[k] for k in alpha.keys() if k < S + 1}       # 1 -> S
    wi = _unwrap("wi", variables)                                   # 1 -> S
    w0 = _unwrap("w0", variables)                                   # 1 -> S
    wtn = _unwrap("wtn", variables)                                 # 1 -> S
    
    assert len(axes) == 3, "axes must have 3 elements"
    
    # Upper ax: velocity and time
    axes[0].plot(list(v.keys()), 3.6 * np.array(list(v.values())), lw=1, ls=ls, label=f"velocity{label_suffix}")
    axes[0].set_ylabel("Velocity (km/h)")
    
    if plot_time:
        ax_ = axes[0].twinx()
        ax_.plot(list(t.keys()), np.cumsum(list(t.values())), lw=1, ls=ls, label=f"time{label_suffix}", color="grey")
        ax_.set_ylabel("Time (s)")
        ax_.legend(loc="lower right")
        axes[0].legend(loc="lower left")
    else:
        axes[0].legend(loc="best")
    
    # Middle ax: traction and braking forces
    axes[1].plot(list(uPlus.keys()), list(uPlus.values()), lw=1, ls=ls, label=f"Traction{label_suffix}")
    axes[1].plot(list(uMinus.keys()), -np.array(list(uMinus.values())), lw=1, ls=ls, label=f"Brake{label_suffix}")
    axes[1].axhline(0, color="grey", linestyle="--", lw=0.5)
    axes[1].set_ylabel("Force (kN)")
    ymax = max(max(uPlus.values()), max(uMinus.values()) if uMinus.values() else 1)
    axes[1].set_ylim(-ymax * 1.1, ymax * 1.1)
    # axes[1].legend(loc="lower center", ncols=3)
    axes[1].legend(loc="best", ncols=3)
    
    # Lower ax: resistances
    axes[2].plot(list(wi.keys()), list(wi.values()), lw=1, ls=ls, label=f"wi{label_suffix}", color="orange")
    axes[2].plot(list(w0.keys()), list(w0.values()), lw=1, ls=ls, label=f"w0{label_suffix}", color="red")
    axes[2].plot(list(wtn.keys()), list(wtn.values()), lw=1, ls=ls, label=f"wtn{label_suffix}", color="blue")
    axes[2].set_ylabel("Resistance (kN)")
    axes[2].legend(loc="best", ncols=3)
    axes[2].set_xlabel(f"Horizontal Intervals ({ds} m)")
    return


def save_vao_plot(
    run_dir: Path, 
    file_prefix: str, 
    file_suffix: str, 
    gcfg: Any, 
    cur_sol: dict, 
    prev_sol: dict | None = None
) -> None:
    """Save VAO result plots in run_dir / {file_prefix}_vao_{file_suffix}.png.

    :param Path run_dir: Directory to save the plot.
    :param str file_prefix: Prefix of the plot file name.
    :param str file_suffix: Suffix of the plot file name.
    :param Any gcfg: Ground configuration.
    :param dict cur_sol: Current solution.
    :param dict | None prev_sol: Previous solution, defaults to None.
        If provided, will draw the old track with dashed line and show 
        differences between new and old tracks.
    """
    fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
    
    draw_vao(
        ax, gcfg, cur_sol, 
        show_ground=True, show_track=True, show_tunnel=True, show_bridge=True, 
        style_track=dict(ls="-", alpha=0.8, lw=0.7, label="Track (new)"),
    )
    
    if prev_sol is not None:
        draw_vao(
            ax, gcfg, prev_sol, 
            show_ground=False, show_track=True, show_tunnel=True, show_bridge=True, 
            style_track=dict(ls="-.", alpha=0.8, lw=0.7, label="Track (old)"),
            style_vpi=dict(marker="x", color="gray", label=None),
        )
        ax2 = ax.twinx()
        ele_diff = [cur_sol["e"][s] - prev_sol['e'][s] for s in range(1, gcfg.S + 2)]
        ax2.plot(range(1, gcfg.S + 2), ele_diff, color="gray", alpha=0.3, label="diff (new - old)")
        ax2.axhline(0, color="grey", alpha=0.3, lw=1, ls=":")
        yma = max(np.abs(ele_diff).max() * 1.1, 1.1e-1)
        ax2.set_ylim(-yma, yma)
        ax2.set_ylabel("Differences (m)")
    
    fig.tight_layout()
    os.makedirs(run_dir / "figs", exist_ok=True)
    fig.savefig(run_dir / f"figs/{file_prefix}_vao_{file_suffix}.png")
    plt.close(fig)
    return
    

def save_tc_plot(run_dir: Path, file_prefix: str, file_suffix: str, SP_vars: dict[str, Any], ti_name: str, ds: float) -> None:
    """Save Train Control results plot in run_dir / {file_prefix}_tc_{ti_name}_{file_suffix}.png.

    :param Path run_dir: Directory to save the plot.
    :param str file_prefix: Prefix of the plot file name.
    :param str file_suffix: Suffix of the plot file name.
    :param dict[str, Any] SP_vars: Optimization variables.
    :param str ti_name: Train instance name.
    :param float ds: Segment length.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 4), dpi=150)
    draw_tc(axes, SP_vars, ds=ds)
    fig.tight_layout()
    os.makedirs(run_dir / "figs", exist_ok=True)
    fig.savefig(run_dir / f"figs/{file_prefix}_tc_{ti_name}_{file_suffix}.png")
    plt.close(fig)
    return