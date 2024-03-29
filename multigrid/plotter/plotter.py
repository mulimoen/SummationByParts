#! /usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import h5py

from argparse import ArgumentParser


def gridlines(obj, x, y):
    for j in range(1, x.shape[0] - 1):
        obj.plot(x[j, :], y[j, :], color="#7f7f7f", linewidth=0.1, alpha=0.3)
    for j in range(1, x.shape[1] - 1):
        obj.plot(x[:, j], y[:, j], color="#7f7f7f", linewidth=0.1, alpha=0.3)

    obj.plot(x[0, :], y[0, :], color="#7f7f7f", linewidth=0.2)
    obj.plot(x[-1, :], y[-1, :], color="#7f7f7f", linewidth=0.2)
    obj.plot(x[:, 0], y[:, 0], color="#7f7f7f", linewidth=0.2)
    obj.plot(x[:, -1], y[:, -1], color="#7f7f7f", linewidth=0.2)


def plot_all(grids, save: bool, filename="figure.png"):
    sym_cmap = plt.get_cmap("PiYG")  # Symmetric around zero
    e_cmap = plt.get_cmap("Greys")

    f, axarr = plt.subplots(2, 2)

    min_rho = min(np.min(g["rho"][-1, :, :]) for g in grids)
    max_rho = max(np.max(g["rho"][-1, :, :]) for g in grids)
    r = 1.2 * max(abs(min_rho - 1), abs(max_rho - 1))
    rho_levels = np.linspace(1 - r, 1 + r, 34)

    min_rhou = min(np.min(g["rhou"][-1, :, :]) for g in grids)
    max_rhou = max(np.max(g["rhov"][-1, :, :]) for g in grids)
    r = 1.2 * max(abs(min_rhou - 1), abs(max_rhou - 1))
    rhou_levels = np.linspace(1 - r, 1 + r, 20)

    min_rhov = min(np.min(g["rhov"][-1, :, :]) for g in grids)
    max_rhov = max(np.max(g["rhov"][-1, :, :]) for g in grids)
    r = 1.2 * max(abs(min_rhov), abs(max_rhov))
    rhov_levels = np.linspace(-r, r, 20)

    min_e = min(np.min(g["e"][-1, :, :]) for g in grids)
    max_e = max(np.max(g["e"][-1, :, :]) for g in grids)
    e_levels = np.linspace(min_e, max_e)

    for g in grids:
        x = g["x"]
        y = g["y"]
        axarr[0, 0].contourf(x, y, g["rho"][-1, :, :], cmap=sym_cmap, levels=rho_levels)
        gridlines(axarr[0, 0], x, y)

        axarr[0, 1].contourf(
            x, y, g["rhou"][-1, :, :], cmap=sym_cmap, levels=rhou_levels
        )
        gridlines(axarr[0, 1], x, y)

        axarr[1, 0].contourf(
            x, y, g["rhov"][-1, :, :], cmap=sym_cmap, levels=rhov_levels
        )
        gridlines(axarr[1, 0], x, y)

        axarr[1, 1].contourf(x, y, g["e"][-1, :, :], cmap=e_cmap, levels=e_levels)
        gridlines(axarr[1, 1], x, y)

    axarr[0, 0].set_title(r"$\rho$")
    axarr[0, 0].set_xlabel("x")
    axarr[0, 0].set_ylabel("y")
    norm = mpl.colors.Normalize(vmin=rho_levels[0], vmax=rho_levels[-1])
    sm = plt.cm.ScalarMappable(cmap=sym_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axarr[0, 0])

    axarr[0, 1].set_title(r"$\rho u$")
    axarr[0, 1].set_xlabel("x")
    axarr[0, 1].set_ylabel("y")
    norm = mpl.colors.Normalize(vmin=rhou_levels[0], vmax=rhou_levels[-1])
    sm = plt.cm.ScalarMappable(cmap=sym_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axarr[0, 1])

    axarr[1, 0].set_title(r"$\rho v$")
    axarr[1, 0].set_xlabel("x")
    axarr[1, 0].set_ylabel("y")
    norm = mpl.colors.Normalize(vmin=rhov_levels[0], vmax=rhov_levels[-1])
    sm = plt.cm.ScalarMappable(cmap=sym_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axarr[1, 0])

    axarr[1, 1].set_title(r"$e$")
    axarr[1, 1].set_xlabel("x")
    axarr[1, 1].set_ylabel("y")
    norm = mpl.colors.Normalize(vmin=e_levels[0], vmax=e_levels[-1])
    sm = plt.cm.ScalarMappable(cmap=e_cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axarr[1, 1])

    if save:
        plt.savefig(filename, bbox_inches="tight", dpi=600)

    plt.show()


def pressure(rho, rhou, rhov, e):
    gamma = 1.4
    return (gamma - 1) * (e - (rhou ** 2 + rhov ** 2) / (2 * rho))


def plot_pressure(grids, save: bool, filename="figure.png"):
    cmap = plt.get_cmap("RdGy")
    Mach = 0.5
    gamma = 1.4

    p = [
        pressure(
            g["rho"][-1, :, :],
            g["rhou"][-1, :, :],
            g["rhov"][-1, :, :],
            g["e"][-1, :, :],
        )
        for g in grids
    ]

    flat_p = np.array([])
    for p_ in p:
        flat_p = np.append(flat_p, p_)

    max_p = np.max(flat_p)
    min_p = np.min(flat_p)

    p_inf = 1 / (gamma * Mach ** 2)

    r = max(max_p - p_inf, p_inf - min_p)

    levels = np.linspace(p_inf - r, p_inf + r, 30)

    for g, p_ in zip(grids, p):
        x = g["x"]
        y = g["y"]

        plt.contourf(x, y, p_, cmap=cmap, levels=levels)
        gridlines(plt, x, y)

    plt.title("Pressure")
    norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)

    plt.xlabel("x")
    plt.ylabel("y")

    if save:
        plt.savefig(filename, bbox_inches="tight", dpi=600)

    plt.show()


def plot_pressure_slider(grids, save: bool, filename="figure.png"):
    cmap = plt.get_cmap("RdGy")
    gamma = 1.4  # Assumption might be wrong
    Mach = 0.5

    def p(itime):
        return [
            pressure(
                g["rho"][itime, :, :],
                g["rhou"][itime, :, :],
                g["rhov"][itime, :, :],
                g["e"][itime, :, :],
            )
            for g in grids
        ]

    max_p = 3.0
    min_p = 1.75
    p_inf = 1 / (gamma * Mach ** 2)
    r = max(max_p - p_inf, p_inf - min_p)
    levels = np.linspace(p_inf - r, p_inf + r, 30)

    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(
        2, 2, figure=fig, width_ratios=[1, 0.02], height_ratios=[1, 0.02]
    )
    ax = fig.add_subplot(gs[0, 0])
    slider_ax = fig.add_subplot(gs[1, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf
    for g in grids:
        x = g["x"]
        xmin = min(xmin, x.min())
        xmax = max(xmax, x.max())
        y = g["y"]
        ymin = min(ymin, y.min())
        ymax = max(ymax, y.max())
        gridlines(ax, x, y)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.title("Pressure")
    norm = mpl.colors.Normalize(vmin=levels[0], vmax=levels[-1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, cax=cbar_ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    itime = len(t) - 1
    slider = mpl.widgets.Slider(
        slider_ax, "itime", 0, itime, valinit=itime, valstep=1, valfmt="%0.0f"
    )

    class Updater(object):
        def __init__(this):
            this.contours = None

        def update(this, itime):
            itime = int(itime)
            for g in grids:
                if this.contours is not None:
                    for coll in this.contours.collections:
                        coll.remove()
                pres = pressure(
                    g["rho"][itime, :, :],
                    g["rhou"][itime, :, :],
                    g["rhov"][itime, :, :],
                    g["e"][itime, :, :],
                )
                this.contours = ax.contourf(
                    g["x"],
                    g["y"],
                    pres,
                    cmap=cmap,
                    levels=levels,
                )
                slider.valtext.set_text(t[itime])

    up = Updater()
    up.update(itime)
    slider.on_changed(up.update)

    plt.show()


def read_from_file(filename):
    grids = []

    file = h5py.File(filename, "r")

    for groupname in file:
        group = file[groupname]
        if not isinstance(group, h5py.Group):
            continue
        grids.append(
            {
                "x": group["x"][:],
                "y": group["y"][:],
                "rho": group["rho"][:],
                "rhou": group["rhou"][:],
                "rhov": group["rhov"][:],
                "e": group["e"][:],
            }
        )

    return grids, file["t"]


def main():
    parser = ArgumentParser(description="Plot a solution from the eulersolver")
    parser.add_argument("filename", metavar="filename", type=str)
    parser.add_argument("-s", help="Save figure", action="store_true", dest="save")
    parser.add_argument(
        "-o",
        help="Output of saved figure",
        type=str,
        default="figure.png",
        dest="output",
    )
    parser.add_argument(
        "-a", help="Show all four variables", action="store_true", dest="all"
    )
    parser.add_argument("--slider", help="Add slider", action="store_true")

    args = parser.parse_args()
    filename = args.filename

    grids, t = read_from_file(filename)

    if args.all:
        plot_all(grids, args.save, args.output)
    else:
        if args.slider:
            plot_pressure_slider(grids, t)
        else:
            plot_pressure(grids, args.save, args.output)


if __name__ == "__main__":
    main()
