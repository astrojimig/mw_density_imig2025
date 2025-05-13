"""
plotting_helpers.py
Helper functions for plotting and making figures

Reference: J. Imig et al. 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib as mpl
import matplotlib.patches as patches
import numpy as np
from mw_density.sample_selection import setup_maap_bins
from typing import Callable
from numpy.typing import NDArray
from matplotlib.axes import Axes

age_bins, mh_bins = setup_maap_bins()


def plot_sun_and_gc(
    size: float = 1,
    r_sun: float = -8.122,
    add_bar: bool = False,
    labels: bool = True,
    ax: Axes = plt.subplot(),
) -> None:
    """Marks the sun and the GC on a plot"""
    # Plot GC
    for i, mc in enumerate(["w", "k"]):
        ax.scatter(
            [0],
            [0],
            marker="+",
            c=mc,
            zorder=12 + i,
            s=(400 - (i * 100)) * size,
            lw=(5 - i * 2) * size,
            snap=False,
        )
    # Plot Sun
    ax.scatter(
        [r_sun],
        [0],
        marker="o",
        c="k",
        facecolor="k",
        s=250 * size,
        zorder=12,
        label="Sun",
        lw=2 * size,
        path_effects=[patheffects.withStroke(linewidth=5, foreground="w")],
    )
    # dot in center for sun symbol
    ax.scatter(
        [r_sun],
        [0],
        marker=".",
        c="w",
        s=50 * size,
        zorder=13,
        snap=False,
    )
    if labels:
        ax.text(
            r_sun,
            1 * size,
            "Sun",
            horizontalalignment="center",
            verticalalignment="bottom",
            zorder=13,
            color="k",
            fontsize=24 * size,
            weight="bold",
            path_effects=[
                patheffects.withStroke(linewidth=5 * size, foreground="w")
            ],
        )
        ax.text(
            0,
            1 * size,
            "Galactic\nCenter",
            horizontalalignment="center",
            verticalalignment="bottom",
            zorder=13,
            color="k",
            fontsize=24 * size,
            weight="bold",
            path_effects=[
                patheffects.withStroke(linewidth=5 * size, foreground="w")
            ],
        )
    if add_bar:  # ellipse for bar location
        bar_angle = 25 if r_sun > 0 else -25
        gbar = patches.Ellipse(
            (0, 0),
            10.0,
            0.4 * 10.0,
            angle=bar_angle,
            facecolor="None",
            edgecolor="k",
        )
        ax.add_patch(gbar)

    return


def plot_model_from_params(
    model: Callable, params: NDArray, savename: str = ""
) -> None:
    r_coords = np.arange(-20, 20, 0.1)
    z_coords = np.arange(-20, 20, 0.1)
    # massdens = model(params, r_coords, z_coords)
    massdens = mass_at_loc2d(model, params, r_coords, z_coords)
    if len(params) == 5:
        hri, hro, hz0, rp, rf = params
        norm = 1.0
    else:
        hri, hro, hz0, rp, rf, norm = params

    plt.figure(figsize=(30, 15))
    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)  # full pix
    ax_r2d = plt.subplot2grid((2, 4), (0, 2), colspan=1, rowspan=1)  # 2D R
    ax_z2d = plt.subplot2grid((2, 4), (0, 3), colspan=1, rowspan=1)  # 2D Z
    ax_r1d = plt.subplot2grid((2, 4), (1, 2), colspan=1, rowspan=1)  # 1D R
    ax_z1d = plt.subplot2grid((2, 4), (1, 3), colspan=1, rowspan=1)  # 1D Z

    # Full Pic
    cim = ax0.imshow(
        np.log10(massdens), extent=(-20, 20, -20, 20), origin="lower"
    )
    plt.colorbar(cim, ax=ax0, label=r"log($\nu_{*}$)")
    ax0.set_xlabel("r (kpc)")
    ax0.set_ylabel("z (kpc)")
    ax0.set_title("Total Profile", fontsize=36)

    for ax in [ax_r2d, ax_z2d]:
        plot_sun_and_gc(size=1, ax=ax, labels=False)

    plot_sun_and_gc(size=1, ax=ax0)

    # R profile 2D
    rprofile = model(params, r_coords, np.zeros(len(r_coords)))
    r2d = np.zeros((len(r_coords), len(z_coords)))
    for ix, x in enumerate(r_coords):
        r = np.sqrt((x**2) + (z_coords**2))
        r2d[ix] += np.interp(r, r_coords, rprofile)

    ax_r2d.imshow(np.log10(r2d.T), extent=(-20, 20, -20, 20), origin="lower")
    ax_r2d.set_title("Radial Profile", fontsize=36)
    ax_r2d.set_xlabel("x (kpc)")
    ax_r2d.set_ylabel("y (kpc)")

    # Z profile 2D
    z2d = massdens
    ax_z2d.imshow(np.log10(z2d), extent=(-20, 20, -20, 20), origin="lower")
    ax_z2d.set_title("Vertical Profile", fontsize=36)
    ax_z2d.set_xlabel("x (kpc)")
    ax_z2d.set_ylabel("z (kpc)")

    for ax in [ax_r1d, ax_z1d]:
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylabel(r"log($\nu_{*}$)")

    for z in [0, 5, 10, 15, 19.9]:
        i = np.where(r_coords >= z)[0][0]
        ax_r1d.scatter(
            r_coords,
            massdens[i],
            c=np.log10(massdens[i]),
            label="r={}".format(r_coords[i]),
        )
        ax_r1d.text(
            0,
            massdens[i][int(len(r_coords) / 2)],
            "z = {} kpc".format(int(r_coords[i])),
            ha="center",
            va="bottom",
        )

    for z in [0, 5, 10, 15, 19.9]:
        i = np.where(r_coords >= z)[0][0]
        ax_z1d.scatter(
            r_coords,
            massdens.T[i],
            c=np.log10(massdens.T[i]),
            label="z={}".format(r_coords[i]),
        )
        ax_z1d.text(
            0,
            massdens.T[i][int(len(r_coords) / 2)],
            "r = {} kpc".format(int(r_coords[i])),
            ha="center",
            va="bottom",
        )

    ax_z1d.set_xlabel("z (kpc)")
    ax_r1d.set_xlabel("r (kpc)")

    plt.tight_layout()
    if savename:
        plt.savefig(savename, bbox_inches="tight")

    plt.show()


def mass_at_loc2d(
    model: Callable,
    params: NDArray,
    r_points: NDArray,
    z_points: NDArray,
) -> NDArray:
    """Combined profile exponential for MW modeling.
    Inputs:
        r: array: array of radii values to evaluate at (kpc)
        Z: array: array of z values to evaluate at (kpc)
        hri: float: inner exponetial parameter
        hro: float: outer exponetial parameter
        hz0: float: scale height (kpc)
        rp: float: radial break radius (kpc)
        rf: float: scale radius for flaring (kpc)
        norm: optional, float: assumed value at Solar Radius (default 1.0)
    Outputs:
        (r)x(z) density profile
    """

    out = np.zeros((len(r_points), len(z_points)))
    # Change dtype
    params = np.array(params).astype("float64")
    r_points = np.array(r_points).astype("float64")
    z_points = np.array(z_points).astype("float64")
    for i in range(len(r_points)):
        out[i] += model(params, r_points, np.ones(len(z_points)) * z_points[i])
    return out


# And some plotting helper functions...
def bin_count_plot(
    lowalph: NDArray, highalph: NDArray, savename: str = ""
) -> None:
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.title(
        "low alpha; <N> = {} / {}".format(
            int(np.mean(lowalph)), int(np.median(lowalph))
        )
    )
    plt.imshow(
        ncount_distmass_low.T,
        aspect=10,
        origin="lower",
        extent=(
            age_bins["min"][0],
            age_bins["max"][-1],
            mh_bins["min"][0],
            mh_bins["max"][-1],
        ),
    )

    for f in mh_bins["min"]:
        plt.axhline(f, c="w")

    fakeages = np.linspace(
        age_bins["min"][0], age_bins["max"][-1], len(age_bins["max"]) + 1
    )
    for a in fakeages:
        plt.axvline(a, c="w")

    plt.xticks(fakeages, np.round(age_bins["center"], 1))
    plt.yticks(mh_bins["center"])
    plt.xlabel("Age")
    plt.ylabel("[Fe/H]")

    for i in range(len(age_bins["center"])):
        for j in range(len(mh_bins["center"])):
            ac = (fakeages[i] + fakeages[i + 1]) / 2
            txt1 = plt.text(
                ac,
                mh_bins["center"][j],
                "{}".format(int(lowalph[i][j])),
                ha="center",
                va="center",
                color="white",
                weight="bold",
            )
            txt1.set_path_effects(
                [patheffects.withStroke(linewidth=3, foreground="k")]
            )

    plt.subplot(122)
    plt.title(
        "high alpha; <N> = {} / {}".format(
            int(np.mean(highalph)), int(np.median(highalph))
        )
    )
    plt.imshow(
        ncount_distmass_high.T,
        aspect=10,
        origin="lower",
        extent=(
            age_bins["min"][0],
            age_bins["max"][-1],
            mh_bins["min"][0],
            mh_bins["max"][-1],
        ),
    )

    for f in mh_bins["min"]:
        plt.axhline(f, c="w")

    fakeages = np.linspace(
        age_bins["min"][0], age_bins["max"][-1], len(age_bins["max"]) + 1
    )
    for a in fakeages:
        plt.axvline(a, c="w")

    plt.xticks(fakeages, np.round(age_bins["center"], 1))
    plt.yticks(mh_bins["center"])
    plt.xlabel("Age")
    plt.ylabel("[Fe/H]")

    for i in range(len(age_bins["center"])):
        for j in range(len(mh_bins["center"])):
            ac = (fakeages[i] + fakeages[i + 1]) / 2
            txt1 = plt.text(
                ac,
                mh_bins["center"][j],
                "{}".format(int(highalph[i][j])),
                ha="center",
                va="center",
                color="white",
                weight="bold",
            )
            txt1.set_path_effects(
                [patheffects.withStroke(linewidth=3, foreground="k")]
            )

    plt.tight_layout()
    if savename:
        plt.savefig(savename, bbox_inches="tight")
    plt.show()


# And some plotting helper functions...
def bin_count_plot_histo(
    lowalph: NDArray,
    highalph: NDArray,
    savename: str = "",
    orientation: str = "horizontal",
) -> None:
    if orientation == "horizontal":
        plt.figure(figsize=(40, 18))
        ax1 = plt.subplot2grid((4, 9), (0, 0), colspan=3, rowspan=1)  # hist1a
        ax2 = plt.subplot2grid((4, 9), (1, 0), colspan=3, rowspan=3)  # smd1
        ax3 = plt.subplot2grid((4, 9), (1, 3), colspan=1, rowspan=3)  # hist1b
        ax4 = plt.subplot2grid((4, 9), (0, 5), colspan=3, rowspan=1)  # hist1a
        ax5 = plt.subplot2grid((4, 9), (1, 5), colspan=3, rowspan=3)  # smd1
        ax6 = plt.subplot2grid((4, 9), (1, 8), colspan=1, rowspan=3)  # hist1b

    elif orientation == "vertical":
        plt.figure(figsize=(17, 40))
        ax1 = plt.subplot2grid((10, 4), (0, 0), colspan=3, rowspan=1)  # hist1a
        ax2 = plt.subplot2grid((10, 4), (1, 0), colspan=3, rowspan=3)  # smd1
        ax3 = plt.subplot2grid((10, 4), (1, 3), colspan=1, rowspan=3)  # hist1b
        ax4 = plt.subplot2grid((10, 4), (5, 0), colspan=3, rowspan=1)  # hist1a
        ax5 = plt.subplot2grid((10, 4), (6, 0), colspan=3, rowspan=3)  # smd1
        ax6 = plt.subplot2grid((10, 4), (6, 3), colspan=1, rowspan=3)  # hist1b

    ax1.set_title(r"Low [$\alpha$/M]", fontsize=64)
    ax4.set_title(r"High [$\alpha$/M]", fontsize=64)

    ax2.imshow(
        lowalph.T,
        aspect=10,
        origin="lower",
        extent=(
            age_bins["min"][0],
            age_bins["max"][-1],
            mh_bins["min"][0],
            mh_bins["max"][-1],
        ),
        cmap=LinearSegmentedColormap.from_list(
            "", ["white", "tab:blue", "darkblue"]
        ),
        vmin=100,
        vmax=np.max(lowalph.T),
    )

    ax5.imshow(
        highalph.T,
        aspect=10,
        origin="lower",
        extent=(
            age_bins["min"][0],
            age_bins["max"][-1],
            mh_bins["min"][0],
            mh_bins["max"][-1],
        ),
        cmap=LinearSegmentedColormap.from_list(
            "", ["white", "tab:red", "darkred"]
        ),
        vmin=100,
        vmax=np.max(highalph.T),
    )

    # Grid lines
    for ax in [ax2, ax5]:
        for f in mh_bins["min"]:
            ax.axhline(f, c="lightgray")
        fakeages = np.linspace(
            age_bins["min"][0], age_bins["max"][-1], len(age_bins["min"]) + 1
        )
        for a in fakeages:
            ax.axvline(a, c="lightgray")

        ac = fakeages
        ax.set_xticks(ac)
        age_tick_labels = [
            round(x, 1)
            for x in np.append(
                np.log10(age_bins["min"] * 1e9),
                np.log10(age_bins["max"][-1] * 1e9),
            )
        ]
        ax.set_xticklabels(age_tick_labels, rotation=90)
        yticks = np.append(mh_bins["min"], mh_bins["max"][-1])
        ax.set_yticks(yticks)
        ax.set_yticklabels(np.round(yticks, 2))
        ax.set_xlabel(r"log$_{10}$(age)", fontsize=48)
        ax.set_ylabel("[M/H]", fontsize=48)
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20

    # Add star count numbers
    for i in range(len(age_bins["center"])):
        for j in range(len(mh_bins["center"])):
            ac = (fakeages[i] + fakeages[i + 1]) / 2
            nlow = int(lowalph[i][j])
            nhigh = int(highalph[i][j])
            if nlow > 0:
                txt1 = ax2.text(
                    ac,
                    mh_bins["center"][j],
                    f"{nlow}",
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                    fontsize=15,
                )
                txt1.set_path_effects(
                    [patheffects.withStroke(linewidth=3, foreground="k")]
                )
            if (nhigh > 0) | (j > 3):
                txt1 = ax5.text(
                    ac,
                    mh_bins["center"][j],
                    f"{nhigh}",
                    ha="center",
                    va="center",
                    color="white",
                    weight="bold",
                    fontsize=15,
                )
                txt1.set_path_effects(
                    [patheffects.withStroke(linewidth=3, foreground="k")]
                )

    histlim1 = 30000
    histlim2 = 20000
    step = np.sum(lowalph, axis=1)
    step = np.append(step[0], step)
    ax1.step(fakeages, step, c="tab:blue", lw=5)

    for ax in [ax1, ax4]:
        ax.set_ylim(0, histlim1)
        tick_values = np.arange(0, histlim1 + 1000, 5000)
        tick_value_labels = [int(ii) for ii in tick_values / 1000]
        tick_value_labels[0] = ""
        ax.set_yticks(tick_values)
        ax.set_yticklabels(tick_value_labels)
        ax.set_xlim(age_bins["min"][0], age_bins["max"][-1])
        ax.set_xticks(fakeages)
        ax.set_xticklabels([])
        ax.set_ylabel(r"N$_{/1000}$", fontsize=36)
        ax.yaxis.labelpad = 20

    for ax in [ax3, ax6]:
        ax.set_xlim(0, histlim2)
        tick_values = np.arange(0, histlim2 + 1000, 5000)
        tick_value_labels = [int(ii) for ii in tick_values / 1000]
        tick_value_labels[0] = ""
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_value_labels)
        ax.set_ylim(mh_bins["min"][0], mh_bins["max"][-1])
        ax.set_yticks(np.append(mh_bins["min"], mh_bins["max"][-1]))
        ax.set_yticklabels([])
        ax.xaxis.tick_top()
        ax.set_xlabel(r"N$_{/1000}$", fontsize=36)
        ax.xaxis.set_label_position("top")
        ax.xaxis.labelpad = 20

    step = np.sum(highalph, axis=1)
    step = np.append(step[0], step)
    ax4.step(fakeages, step, c="tab:red", lw=5)
    # ax4.set_ylim(0,histlim2)
    # ax4.set_yticks(np.arange(0,histlim2+1000,5000))
    # ax4.set_yticklabels(['','5','10','15','20','25','30'])
    step = np.sum(lowalph, axis=0)
    step = np.append(step, step[-1])
    ax3.step(
        step, np.append(mh_bins["min"], mh_bins["max"][-1]), c="tab:blue", lw=5
    )

    step = np.sum(highalph, axis=0)
    step = np.append(step, step[-1])
    ax6.step(
        step, np.append(mh_bins["min"], mh_bins["max"][-1]), c="tab:red", lw=5
    )

    for ax in [ax1, ax3, ax4, ax6]:
        ax.grid()

    # annotation text
    ax5.axhline(-0.7, lw=6, c="k", linestyle=":")
    txt1 = ax5.text(
        fakeages[0], -0.72, "  Assumed Ages", va="top", ha="left", color="k"
    )
    txt1.set_path_effects(
        [patheffects.withStroke(linewidth=20, foreground="w")]
    )
    ax2.axhline(-0.7, lw=6, c="k", linestyle=":")
    ax2.text(
        fakeages[0], -0.72, "  [M/H] limit", va="top", ha="left", color="k"
    )

    plt.subplots_adjust(wspace=0, hspace=0)

    # Add isophots
    x = fakeages
    y = np.append(mh_bins["min"], mh_bins["max"][-1])
    z = lowalph.T
    add_iso_line(ax2, 100, "gray", x, y, z)
    z = highalph.T
    add_iso_line(ax5, 100, "gray", x, y, z)

    if savename:
        plt.savefig(savename, bbox_inches="tight")

    plt.show()
    return


def add_iso_line(
    ax: Axes, value: int, color: str, x: NDArray, y: NDArray, z: NDArray
) -> None:
    """Add isophot line to show which bins have enough stars to get MCMC'd"""
    v = np.diff(z > value, axis=1)
    h = np.diff(z > value, axis=0)

    lineval = np.argwhere(v.T)
    vlines = np.array(
        list(
            zip(
                np.stack((x[lineval[:, 0] + 1], y[lineval[:, 1]])).T,
                np.stack((x[lineval[:, 0] + 1], y[lineval[:, 1] + 1])).T,
            )
        )
    )
    lineval = np.argwhere(h.T)
    hlines = np.array(
        list(
            zip(
                np.stack((x[lineval[:, 0]], y[lineval[:, 1] + 1])).T,
                np.stack((x[lineval[:, 0] + 1], y[lineval[:, 1] + 1])).T,
            )
        )
    )
    lines = np.vstack((vlines, hlines))
    ax.add_collection(LineCollection(lines, lw=5, colors=color))


def bin_count_plot_histo_old(
    lowalph: NDArray,
    highalph: NDArray,
    savename: str = "",
    orientation: str = "horizontal",
) -> None:
    if orientation == "horizontal":
        plt.figure(figsize=(40, 14.8))
        ax1 = plt.subplot2grid((4, 9), (0, 0), colspan=3, rowspan=1)  # hist1a
        ax2 = plt.subplot2grid((4, 9), (1, 0), colspan=3, rowspan=3)  # smd1
        ax3 = plt.subplot2grid((4, 9), (1, 3), colspan=1, rowspan=3)  # hist1b
        ax4 = plt.subplot2grid((4, 9), (0, 5), colspan=3, rowspan=1)  # hist1a
        ax5 = plt.subplot2grid((4, 9), (1, 5), colspan=3, rowspan=3)  # smd1
        ax6 = plt.subplot2grid((4, 9), (1, 8), colspan=1, rowspan=3)  # hist1b

    elif orientation == "vertical":
        plt.figure(figsize=(17, 40))
        ax1 = plt.subplot2grid((10, 4), (0, 0), colspan=3, rowspan=1)  # hist1a
        ax2 = plt.subplot2grid((10, 4), (1, 0), colspan=3, rowspan=3)  # smd1
        ax3 = plt.subplot2grid((10, 4), (1, 3), colspan=1, rowspan=3)  # hist1b
        ax4 = plt.subplot2grid((10, 4), (5, 0), colspan=3, rowspan=1)  # hist1a
        ax5 = plt.subplot2grid((10, 4), (6, 0), colspan=3, rowspan=3)  # smd1
        ax6 = plt.subplot2grid((10, 4), (6, 3), colspan=1, rowspan=3)  # hist1b

    ax1.set_title(r"Low [$\alpha$/M]", fontsize=64)
    ax4.set_title(r"High [$\alpha$/M]", fontsize=64)

    ax2.imshow(
        lowalph.T,
        aspect=10,
        origin="lower",
        extent=[
            age_bins["min"][0],
            age_bins["max"][-1],
            mh_bins["min"][0],
            mh_bins["max"][-1],
        ],
        cmap=LinearSegmentedColormap.from_list(
            "", ["white", "tab:blue", "darkblue"]
        ),
        vmin=1,
        vmax=np.max(lowalph.T),
    )

    ax5.imshow(
        highalph.T,
        aspect=10,
        origin="lower",
        extent=[
            age_bins["min"][0],
            age_bins["max"][-1],
            mh_bins["min"][0],
            mh_bins["max"][-1],
        ],
        cmap=LinearSegmentedColormap.from_list(
            "", ["white", "tab:red", "darkred"]
        ),
        vmin=1,
        vmax=np.max(highalph.T),
    )

    for ax in [ax2, ax5]:
        for f in mh_bins["min"]:
            ax.axhline(f, c="lightgray")
        fakeages = np.linspace(
            age_bins["min"][0], age_bins["max"][-1], len(age_bins["max"]) + 1
        )
        for a in fakeages:
            ax.axvline(a, c="lightgray")

        ac = fakeages
        ax.set_xticks(ac)
        age_tick_labels = [
            round(x, 1)
            for x in np.append(
                np.log10(age_bins["min"] * 1e9),
                np.log10(age_bins["max"][-1] * 1e9),
            )
        ]
        ax.set_xticklabels(age_tick_labels, rotation=90)
        ax.set_yticks(np.append(mh_bins["min"], mh_bins["max"][-1]))
        ax.set_yticklabels(
            np.round(np.append(mh_bins["min"], mh_bins["max"][-1]), 2)
        )
        ax.set_xlabel(r"log$_{10}$(age)", fontsize=48)
        ax.set_ylabel("[M/H]", fontsize=48)
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20

    for i in range(len(age_bins["center"])):
        for j in range(len(mh_bins["center"])):
            ac = (fakeages[i] + fakeages[i + 1]) / 2
            txt1 = ax2.text(
                ac,
                mh_bins["center"][j],
                "{}".format(int(lowalph[i][j])),
                ha="center",
                va="center",
                color="white",
                weight="bold",
                fontsize=15,
            )
            txt1.set_path_effects(
                [patheffects.withStroke(linewidth=3, foreground="k")]
            )
            txt1 = ax5.text(
                ac,
                mh_bins["center"][j],
                "{}".format(int(highalph[i][j])),
                ha="center",
                va="center",
                color="white",
                weight="bold",
                fontsize=15,
            )
            txt1.set_path_effects(
                [patheffects.withStroke(linewidth=3, foreground="k")]
            )

    histlim1 = 25000
    step = np.sum(lowalph, axis=1)
    step = np.append(step[0], step)
    ax1.step(fakeages, step, c="tab:blue", lw=5)

    for ax in [ax1, ax4]:
        ax.set_ylim(0, histlim1)
        tick_values = np.arange(0, histlim1 + 1000, 5000)
        tick_value_labels = [int(ii) for ii in tick_values / 1000]
        tick_value_labels[0] = ""
        ax.set_yticks(tick_values)
        ax.set_yticklabels(tick_value_labels)

    for ax in [ax3, ax6]:
        ax.set_xlim(0, histlim1)
        ax.set_xticks(tick_values)
        ax.set_xticklabels(tick_value_labels)

    step = np.sum(highalph, axis=1)
    step = np.append(step[0], step)
    ax4.step(fakeages, step, c="tab:red", lw=5)
    # ax4.set_ylim(0,histlim2)
    # ax4.set_yticks(np.arange(0,histlim2+1000,5000))
    # ax4.set_yticklabels(['','5','10','15','20','25','30'])
    step = np.sum(lowalph, axis=0)
    step = np.append(step, step[-1])
    ax3.step(
        step, np.append(mh_bins["min"], mh_bins["max"][-1]), c="tab:blue", lw=5
    )

    step = np.sum(highalph, axis=0)
    step = np.append(step, step[-1])
    ax6.step(
        step, np.append(mh_bins["min"], mh_bins["max"][-1]), c="tab:red", lw=5
    )

    for ax in [ax1, ax3, ax4, ax6]:
        ax.grid()

    for ax in [ax1, ax4]:
        ax.set_xlim(age_bins["min"][0], age_bins["max"][-1])
        ax.set_xticks(fakeages)
        ax.set_xticklabels([])
        ax.set_ylabel(r"N$_{/1000}$", fontsize=36)
        ax.yaxis.labelpad = 20
        # ax.axhline(0,c='k',lw=5)

    for ax in [ax3, ax6]:
        ax.set_ylim(mh_bins["min"][0], mh_bins["max"][-1])
        ax.set_yticks(np.append(mh_bins["min"], mh_bins["max"][-1]))
        ax.set_yticklabels([])
        # ax.axvline(0,c='k',lw=5)
        ax.xaxis.tick_top()
        ax.set_xlabel(r"N$_{/1000}$", fontsize=36)
        ax.xaxis.set_label_position("top")
        ax.xaxis.labelpad = 20

    plt.subplots_adjust(wspace=0, hspace=0)

    if savename:
        plt.savefig(savename, bbox_inches="tight")
    plt.show()
    return


def plot_model_from_params_ex(
    model: Callable,
    params: list,
    density_cmap: str = "magma",
    savename: str = "",
    annotate: bool = True,
    bin_title: str = "",
) -> None:
    density_cmap = "magma"
    dmin, dmax = (-30, 2)

    params = np.array(params).astype("float64")
    r_coords = np.arange(-20, 20, 0.1)
    z_coords = np.arange(-20, 20, 0.1)
    r_coords = r_coords.astype("float64")
    z_coords = z_coords.astype("float64")

    # massdens = model(params, r_coords, z_coords)
    massdens = mass_at_loc2d(model, params, r_coords, z_coords)
    if len(params) == 5:
        hri, hro, hz0, rp, rf = params
        norm = 1.0
    else:
        hri, hro, hz0, rp, rf, norm = params

    plt.figure(figsize=(40, 20))
    sm_size = 4
    aspect_size = ((sm_size * 2 + 1), (sm_size * 2 + 1) * 2)

    ax0 = plt.subplot2grid(
        aspect_size, (0, 0), colspan=aspect_size[0] + 1, rowspan=aspect_size[0]
    )  # full pix
    # ax0_cax = plt.subplot2grid(aspect_size, (0, aspect_size[0]), colspan=1, rowspan=aspect_size[0])

    ax_r2d = plt.subplot2grid(
        aspect_size, (0, aspect_size[0] + 1), colspan=sm_size, rowspan=sm_size
    )  # 2D R
    ax_z2d = plt.subplot2grid(
        aspect_size,
        (0, aspect_size[0] + sm_size + 1),
        colspan=sm_size,
        rowspan=sm_size,
    )  # 2D Z

    ax_r1d = plt.subplot2grid(
        aspect_size,
        (sm_size, aspect_size[0] + 1),
        colspan=sm_size,
        rowspan=sm_size,
    )  # 1D R
    ax_z1d = plt.subplot2grid(
        aspect_size,
        (sm_size, aspect_size[0] + sm_size + 1),
        colspan=sm_size,
        rowspan=sm_size,
    )  # 1D Z

    ax_Rc = plt.subplot2grid(
        aspect_size,
        (aspect_size[0] - 1, aspect_size[0] + 1),
        rowspan=1,
        colspan=sm_size,
    )  # 1D R#
    ax_Zc = plt.subplot2grid(
        aspect_size,
        (aspect_size[0] - 1, aspect_size[0] + sm_size + 1),
        colspan=sm_size,
        rowspan=1,
    )  # 1D R

    # Full Pic
    cim = ax0.imshow(
        np.log10(massdens),
        extent=(-20, 20, -20, 20),
        origin="lower",
        cmap=density_cmap,
        vmax=dmax,
        vmin=dmin,
    )
    plt.colorbar(cim, ax=ax0, extend="both").set_label(
        label=r"log($\nu_{*}$)", size=36
    )
    ax0.set_xlabel("x (kpc)")
    ax0.set_ylabel("z (kpc)")
    ax0.set_title("Total Density Profile", fontsize=36)
    for ax in [ax_r2d, ax_z2d]:
        plot_sun_and_gc(size=1, ax=ax, labels=False)

    plot_sun_and_gc(size=1.5, ax=ax0)

    # R profile 2D
    rprofile = model(params, r_coords, np.zeros(len(r_coords)))
    r2d = np.zeros((len(r_coords), len(z_coords)))
    for ix, x in enumerate(r_coords):
        r = np.sqrt((x**2) + (z_coords**2))
        r2d[ix] += np.interp(r, r_coords, rprofile)

    ax_r2d.imshow(
        np.log10(r2d.T),
        extent=(-20, 20, -20, 20),
        origin="lower",
        cmap=density_cmap,
    )
    ax_r2d.set_title("Radial Profile", fontsize=36)
    ax_r2d.set_xlabel("x (kpc)")
    ax_r2d.set_ylabel("y (kpc)")

    # Z profile 2D
    z2d = np.array([massdens[i] / rprofile for i in range(len(massdens))])
    ax_z2d.imshow(
        np.log10(z2d),
        extent=(-20, 20, -20, 20),
        origin="lower",
        cmap=density_cmap,
        vmax=dmax,
        vmin=dmin,
    )
    ax_z2d.set_title("Vertical Profile", fontsize=36)
    ax_z2d.set_xlabel("x (kpc)")
    ax_z2d.set_ylabel("z (kpc)")

    for ax in [ax_r1d, ax_z1d]:
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylabel(r"$\nu_{*}$")
        ax.set_xlim(-20, 20)

    z_samp = [0, 1, 2, 3, 4]
    cim = ax_z1d.scatter(
        np.ones(len(z_samp)) * -25,
        np.ones(len(z_samp)),
        c=z_samp,
        cmap="binary",
    )
    plt.colorbar(
        cim,
        cax=ax_Rc,
        label=r"z (kpc)",
        orientation="horizontal",
        ticks=z_samp,
    )
    for z in z_samp:
        i = np.where(r_coords >= z)[0][0]
        ax_r1d.plot(
            r_coords,
            massdens[i],
            # c=np.log10(massdens[i]),label='r={}'.format(r_coords[i]))
            c="k",
            lw=6,
        )
        ax_r1d.plot(
            r_coords,
            massdens[i],
            # c=np.log10(massdens[i]),label='r={}'.format(r_coords[i]))
            c=mpl.cm.binary(z / np.max(z_samp)),
            lw=4,
        )
        # ax_r1d.text(0,massdens[i][int(len(r_coords)/2)], 'z = {} kpc'.format(int(r_coords[i])),
        #     ha='center', va='bottom')

    r_samp = [0, 5, 10, 15, 19.9]
    cim = ax_z1d.scatter(
        np.ones(len(r_samp)) * -25,
        np.ones(len(r_samp)),
        c=[0, 5, 10, 15, 20],
        cmap="binary",
    )
    plt.colorbar(
        cim,
        cax=ax_Zc,
        label=r"r (kpc)",
        orientation="horizontal",
        ticks=[0, 5, 10, 15, 20],
    )
    for z in r_samp:
        i = np.where(r_coords >= z)[0][0]
        ax_z1d.plot(r_coords, massdens.T[i], c="k", lw=6)
        ax_z1d.plot(
            r_coords, massdens.T[i], c=mpl.cm.binary(z / np.max(r_samp)), lw=4
        )
        # c=np.log10(massdens.T[i]),label='z={}'.format(r_coords[i]))
        # ax_z1d.text(0,massdens.T[i][int(len(r_coords)/2)], 'r = {} kpc'.format(int(r_coords[i])),
        #     ha='center', va='bottom')

    ax_z1d.set_xlabel("z (kpc)")
    ax_r1d.set_xlabel("r (kpc)")

    ax_r2d.text(3, -18, "z = 0 kpc", color="w")
    ax_z2d.text(3, -18, "y = 0 kpc", color="w")

    paramstring = r"$h_{r, in}$ = " + str(round(1 / params[0], 1)) + " kpc \n"
    paramstring += (
        r"$h_{r, out}$ = " + str(round(1 / params[1], 1)) + " kpc \n"
    )
    paramstring += r"$r_{break}$ = " + str(round(params[3], 1)) + " kpc \n"

    txt1 = ax0.text(
        -18,
        -20,
        paramstring,
        color="w",
        fontsize=48,
        verticalalignment="bottom",
    )
    # txt1.set_path_effects([patheffects.withStroke(linewidth=2, foreground='k')])

    paramstring = (
        r"$h_{z}(R_{\odot})$ = " + str(round(params[2], 1)) + " kpc \n"
    )
    paramstring += r"$A_{flare}$ = " + str(round(params[4], 2)) + " \n"
    paramstring += (
        r"log($\nu_{*,\odot}$) = " + str(round(params[5], 1)) + "\n"
    )  # + r' kpc$^{-3}$ '
    ax0.text(
        0, -20, paramstring, color="w", fontsize=48, verticalalignment="bottom"
    )

    if annotate:
        ax_r1d.text(1, 10**2.5, r"$h_{r, in}^{-1}$", rotation=-20, fontsize=30)
        ax_r1d.text(
            12, 10**-1, r"$h_{r, out}^{-1}$", rotation=-45, fontsize=30
        )

        ax_r1d.set_ylim(10**-12, 10**5)
        for h in [50, 100, 500]:
            ax_r1d.scatter(10, h, c="k", marker="|", s=200, lw=3)
        ax_r1d.scatter(10, 20, c="k", marker="v", s=100)
        ax_r1d.text(
            9,
            10**3,
            r"$r_{break}$",
            verticalalignment="bottom",
            horizontalalignment="left",
            fontsize=30,
        )

        ax_z1d.text(
            7, 10**-22, r"$h_{z}^{-1}(r=20)$", rotation=-50, fontsize=30
        )
        ax_z1d.text(
            -1, 10**-20, r"$h_{z}^{-1}(r=0)$", rotation=-75, fontsize=30
        )
        ax_z1d.set_ylim(10**-40, 10**5)

    # ax_z1d.text(0,10**6, '$h_{z}(R) = h_{z\odot} + A_{flare}(R-R_{\odot})$', fontsize=22,
    #            horizontalalignment='center')
    if not bin_title == None:
        ax0.text(
            -18, 18, bin_title, color="w", fontsize=48, verticalalignment="top"
        )
    else:
        ax0.text(
            -18,
            18,
            "Example Model",
            color="w",
            fontsize=48,
            verticalalignment="top",
        )

    plt.tight_layout()
    if savename:
        plt.savefig(savename, bbox_inches="tight")

    plt.show()
    return


def plot_spiral_arms(ax: Axes, rotate: float) -> None:
    # SPIRAL ARMS
    # https://iopscience.iop.org/article/10.3847/1538-4357/ab4a11/pdf
    perseus = [-23, 115, 40, 8.87, 10.3, 8.7, "green", "Perseus", 2, 5.25]
    sagcar = [2, 97, 24, 6.04, 17.1, 1.0, "magenta", "Sag-Car", -1.5, 5.75]
    local = [-8, 34, 9, 8.26, 11.4, 11.4, "cyan", "Local", -8.2, 3]
    norma = [5, 54, 18, 4.46, -1, 19.5, "red", "Norma", -3.5, 1]
    sctcen = [0, 109, 23, 4.91, 13.1, 12.1, "blue", "Scut-Cen", 0, 1.7]
    outer = [-16, 71, 18, 12.24, 3.0, 9.4, "red", "Outer", -5, 9.75]

    for spiralarm in [perseus, sagcar, local, norma, sctcen, outer]:
        b = np.deg2rad(np.linspace(spiralarm[0], spiralarm[1], 100))
        bkink = np.deg2rad(spiralarm[2] + rotate)
        rkink = spiralarm[3]
        pitch1 = np.deg2rad(spiralarm[4])
        pitch2 = np.deg2rad(spiralarm[5])
        r1 = -1 * (b[b <= bkink] - bkink) * np.tan(pitch1)
        r2 = -1 * (b[b > bkink] - bkink) * np.tan(pitch2)
        r = rkink * np.exp(np.append(r1, r2))
        x = -1 * r * np.cos(b)
        y = r * np.sin(b)
        ax.plot(x, y, c=spiralarm[6], lw=4)

        lastangle = np.rad2deg(np.arctan((y[-2] - y[-1]) / (x[-2] - x[-1])))
        ax.text(
            spiralarm[8],
            spiralarm[9],
            spiralarm[7],
            color=spiralarm[6],
            fontsize=16,
            ha="left",
            va="bottom",
            rotation=lastangle,
        )

        b = np.deg2rad(np.linspace(spiralarm[0] - 60, spiralarm[1] + 60, 100))
        bkink = np.deg2rad(spiralarm[2] + rotate)
        rkink = spiralarm[3]
        pitch1 = np.deg2rad(spiralarm[4])
        pitch2 = np.deg2rad(spiralarm[5])
        r1 = -1 * (b[b < bkink] - bkink) * np.tan(pitch1)
        r2 = -1 * (b[b > bkink] - bkink) * np.tan(pitch2)
        r = rkink * np.exp(np.append(r1, r2))
        x = -1 * r * np.cos(b)
        y = r * np.sin(b)
        ax.plot(x, y, c=spiralarm[6], linestyle="--")
        return


def setup_bin_axes() -> None:
    plt.figure(figsize=(25, 17))
    ax1c = plt.subplot2grid((6, 2), (0, 0), colspan=2)  # cax
    ax1a = plt.subplot2grid((6, 2), (1, 0), rowspan=5)  # high alph
    ax1b = plt.subplot2grid((6, 2), (1, 1), rowspan=5)  # low alph

    ax1a.set_title(r"Low [$\alpha$/M]", fontsize=48)
    ax1b.set_title(r"High [$\alpha$/M]", fontsize=48)

    for ax in [ax1a, ax1b]:
        ax.set_xticks(np.arange(len(age_bins["center"]) + 1) - 0.5)
        ax.set_yticks(np.arange(len(mh_bins["min"]) + 1) - 0.5)
        ax.set_yticklabels(
            [
                round(a, 2)
                for a in np.append(mh_bins["min"], mh_bins["max"][-1])
            ]
        )
        age_tick_labels = [
            round(x, 1)
            for x in np.append(
                np.log10(age_bins["min"] * 1e9),
                np.log10(age_bins["max"][-1] * 1e9),
            )
        ]
        ax.set_xticklabels(age_tick_labels, rotation=90)
        ax.set_ylim(-0.5, len(mh_bins["max"]) - 0.5)
        ax.set_xlim(-0.5, len(age_bins["max"]) - 0.5)
        ax.set_xlabel("log(age)", fontsize=48)
        ax.set_ylabel("[M/H]", fontsize=48)

        for i in range(len(age_bins["center"])):
            ax.axvline(i - 0.5, c="lightgray")
        for i in range(len(mh_bins["center"])):
            ax.axhline(i - 0.5, c="lightgray")

    return ax1a, ax1b, ax1c


# =====================
