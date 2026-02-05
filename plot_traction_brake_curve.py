# root/scripts/plot_traction_brake_curve.py
import numpy as np
import matplotlib

matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from utils import latexify

figsize=(4,4)
dpi=150
legend_fontsize=8
SAVE_FIG = True


def draw_constraints_min(ax, constraints, x, y_max):
    ys = [np.ones_like(x) * y_max]
    ys_labels = ["Max Force Constant"]
    for constraint in constraints:
        params = constraint["params"]
        if constraint["type"] == "linear":
            for i in range(len(params)):
                ys.append(params[i]["k"] * x + params[i]["b"])
                ys_labels.append(f"Linear Constr. {i+1}")
        elif constraint["type"] == "inverse":
            for i in range(len(params)):
                ys.append(params[i]["a"] / x + params[i]["c"])
                ys_labels.append(f"Invers. Constr. {i+1}")
    for y_, y_label in zip(ys, ys_labels):
        ax.plot(x, y_, lw=1.5, alpha=0.5, label=y_label)
    y = np.vstack((ys)).min(axis=0)
    ax.plot(x, y, lw=3, alpha=0.5, label="Min of Constraints")


crh380al_traction = np.array(
    [
        [0.00, 520000.00],
        [1.39, 520000.00],
        [2.78, 520000.00],
        [4.17, 520000.00],
        [5.56, 520000.00],
        [8.33, 520000.00],
        [11.11, 520000.00],
        [13.89, 520000.00],
        [16.67, 513000.00],
        [19.44, 506000.00],
        [22.22, 499000.00],
        [25.00, 492100.00],
        [27.78, 485100.00],
        [30.56, 478100.00],
        [33.33, 471100.00],
        [36.11, 464100.00],
        [38.89, 457100.00],
        [41.67, 450100.00],
        [46.39, 418600.00],
        [47.22, 411200.00],
        [50.00, 388400.00],
        [52.78, 367900.00],
        [55.56, 349500.00],
    ]
)
crh380al_brake = np.array(
    [
        [0.00, 509600.00],
        [1.39, 509600.00],
        [2.78, 509600.00],
        [4.17, 509600.00],
        [5.56, 509600.00],
        [8.33, 509600.00],
        [11.11, 509600.00],
        [13.89, 509600.00],
        [16.67, 509600.00],
        [19.44, 509600.00],
        [22.22, 504600.00],
        [25.00, 499700.00],
        [27.78, 494700.00],
        [30.56, 489700.00],
        [33.33, 484800.00],
        [36.11, 479800.00],
        [38.89, 474800.00],
        [41.67, 469900.00],
        [44.44, 464900.00],
        [47.22, 460000.00],
        [50.00, 455000.00],
        [52.78, 450000.00],
        [55.56, 445100.00],
    ]
)


hxd1d_keys = [
    "v(m/s)",
    "F(traction)(kN)",
    "F(reg brake)(kN)",
    "F(pn brake)(kN)",
]
hxd1d_data = np.array(
    [
        [0.00, 828.47, 420.00, 1491.00],
        [1.39, 827.30, 420.00, 1391.97],
        [1.39, 827.29, 420.00, 1391.83],
        [2.78, 823.96, 420.00, 1333.56],
        [5.56, 811.27, 420.00, 1259.47],
        [8.33, 791.42, 420.00, 1210.97],
        [11.11, 765.43, 420.00, 1175.60],
        [13.89, 734.32, 420.00, 1148.34],
        [16.67, 699.11, 420.00, 1126.55],
        [19.44, 660.82, 420.00, 1108.68],
        [22.22, 620.47, 420.00, 1093.74],
        [25.00, 579.08, 420.00, 1081.05],
        [27.78, 537.67, 420.00, 1070.13],
        [30.56, 497.26, 420.00, 1060.63],
        [33.33, 458.87, 420.00, 1052.29],
        [34.17, 447.90, 420.00, 1049.98],
        [36.11, 423.52, 400.25, 1044.90],
        [38.89, 392.23, 373.77, 1038.31],
        [41.67, 366.02, 347.29, 1032.40],
        [44.44, 345.91, 320.81, 1027.06],
    ]
)


def plot_brake_c():  # CRH380AL
    x = np.linspace(0.001, 200 / 3.6, 1000)

    constraints = [
        {
            "type": "linear",
            "formula": "F <= k * v + b. F(kN). v(m/s)",
            "params": [{"k": -1.787, "b": 544.34}],
        },
        {
            "type": "inverse",
            "formula": "F <= a / v + c. F(kN). v(m/s)",
            "params": [{"a": 42942.9, "c": -121.0}, {"a": 33033.0, "c": 0.0}],
        },
    ]
    y_max = 509.6

    latexify()
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize)

    draw_constraints_min(ax, constraints, x, y_max)

    ax.plot(
        crh380al_brake[:, 0],
        crh380al_brake[:, 1] / 1000,
        lw=4,
        color="red",
        alpha=0.7,
        label="Designed Max Force",
    )
    ax.set_ylim(250, 550)
    ax.set_xlabel(r"$v$ (m/s)")
    ax.set_ylabel("Brake Force (kN)")
    ax.grid(True)
    ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    if SAVE_FIG:
        fig.savefig("crh380al_brake_curve.pdf", dpi=200, bbox_inches="tight")
    plt.show()
    return


def plot_traction_c():  # CRH380AL
    x = np.linspace(0.001, 200 / 3.6, 1000)

    constraints = [
        {
            "type": "linear",
            "formula": "F <= k * v + b. F(kN). v(m/s)",
            "params": [{"k": -2.52, "b": 555.0}, {"k": -6.6737, "b": 728.194}],
        },
        {
            "type": "inverse",
            "formula": "F <= a / v + c. F(kN). v(m/s)",
            "params": [{"a": 19418.1, "c": 0.0}],
        },
    ]
    y_max = 520

    latexify()
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize)
    draw_constraints_min(ax, constraints, x, y_max)

    ax.plot(
        crh380al_traction[:, 0],
        crh380al_traction[:, 1] / 1000,
        lw=4,
        color="red",
        alpha=0.7,
        label="Designed Max Force",
    )
    ax.set_ylim(300, 550)
    ax.set_xlabel(r"$v$ (m/s)")
    ax.set_ylabel("Traction Force (kN)")
    ax.grid(True)
    ax.legend(fontsize=legend_fontsize)
    plt.tight_layout()
    if SAVE_FIG:
        fig.savefig("crh380al_traction_curve.pdf", dpi=200, bbox_inches="tight")
    plt.show()
    return



def plot_brake_h1():
    y_max = 420
    constraints = [
        {
            "type": "inverse",
            "formula": "F <= a / v + c. F(kN). v(m/s)",
            "params": [{"a": 14300, "c": 2.5211}],
        }
    ]
    x = np.linspace(0.001, 160 / 3.6, 1000)

    latexify()
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize)
    draw_constraints_min(ax, constraints, x, y_max)
    ax.plot(hxd1d_data[:, 0], hxd1d_data[:, 2], lw=3, alpha=0.7, label="Designed Max Force")
    ax.set_ylim(100, 450)
    ax.set_xlabel(r"$v$ (m/s)")
    ax.set_ylabel("Braking Force (kN)")
    ax.grid(True)
    ax.legend(ncols=2, fontsize=legend_fontsize)
    plt.tight_layout()
    if SAVE_FIG:
        fig.savefig("hxd1d_brake_curve.pdf", dpi=200, bbox_inches="tight")
    plt.show()
    return


def plot_traction_h1():
    y_max = 827.3
    constraints = [
        {
            "type": "linear",
            "formula": "F <= k * v + b. F(kN). v(m/s)",
            "params": [
                {"k": -2.4072, "b": 830.64},
                {"k": -4.5684, "b": 836.65},
                {"k": -7.146, "b": 850.97},
                {"k": -10.278, "b": 877.92},
            ],
        },
        {
            "type": "inverse",
            "formula": "F <= a / v + c. F(kN). v(m/s)",
            "params": [
                # --- a lighter version
                # { "a": 3531.9, "c": 485.13 },
                # { "a": 8677.1, "c": 225.03 },
                # { "a": 15300, "c": 0 },
                # --- a more accurate version
                {"a": 4467.2, "c": 431.08},
                {"a": 6276.7, "c": 338.02},
                {"a": 8278, "c": 247.96},
                {"a": 10352, "c": 164.98},
                {"a": 12348, "c": 93.158},
                {"a": 15300, "c": 0},
            ],
        },
    ]
    x = np.linspace(0.001, 160 / 3.6, 1000)
    latexify()
    fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=figsize)
    draw_constraints_min(ax, constraints, x, y_max)
    ax.plot(hxd1d_data[:, 0], hxd1d_data[:, 1], lw=3, alpha=0.7, label="Designed Max Force")
    ax.set_ylim(0, 900)
    ax.set_xlabel(r"$v$ (m/s)")
    ax.set_ylabel("Traction Force (kN)")
    ax.grid(True)
    ax.legend(ncols=2, fontsize=legend_fontsize)
    plt.tight_layout()
    if SAVE_FIG:
        fig.savefig("hxd1d_traction_curve.pdf", dpi=200, bbox_inches="tight")
    plt.show()
    return


if __name__ == "__main__":
    plot_brake_c()
    plot_traction_c()
    plot_brake_h1()
    plot_traction_h1()
    pass
