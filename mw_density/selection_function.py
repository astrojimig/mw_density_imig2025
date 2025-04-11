"""
selection_function.py
Code and helper functions for calculating the APOGEE
raw and effective selection functions

Reference: J. Imig et al. 2025"""

import dill as pickle
import numpy as np
from mw_density.sample_selection import distmod_bins, set_env_variables
from astropy import units, coordinates
from typing import Any
import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

set_env_variables()
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SelectionFunction:
    def __init__(self) -> None:
        # Set distmod bins from sample selection
        self.distmod_bins = distmod_bins()
        self.ndistmods, self.distances, self.distmods, self.minmax_distmods = (
            self.distmod_bins
        )
        # Load in files
        # Raw Selection Function
        self.rawsel = self.load_raw_selfunc()
        # Effective Selection Function
        self.effsel = self.load_effselxarea_allbins()
        # self.effsel = self.load_effsel_allbins()
        self.effsel_noarea = self.load_effsel_allbins()

        # Calculate coordinates for effsel
        self.coordinates = self.calculate_coords()

        # Caclulate some volumes
        self.total_volume = self.calc_survey_volume()

    def load_raw_selfunc(self) -> Any:
        """Load in the raw selection function file"""
        rawsel_path = os.path.join(
            PKG_ROOT, "data/selfuncs/apogeeCombinedSF.dat"
        )
        with open(rawsel_path, "rb") as f:
            rawsel = pickle.load(f)
        return rawsel

    def load_effselxarea_allbins(self) -> Any:
        """Load in the effective selection function file for all bins"""
        # Load file
        effsel_name = os.path.join(
            PKG_ROOT, "data/selfuncs/effsel_allbins.npz"
        )
        effsel = np.load(effsel_name)["arr_0"]

        # Check to make sure most bins are valid
        goodbins = 0
        for i in range(len(effsel)):
            test = len(effsel[i][(effsel[i] != 0) & (np.isfinite(effsel[i]))])
            # print(test)
            if test >= 0.5 * (
                len(effsel[i].flatten())
            ):  # more than 90% of effsel is good
                goodbins += 1

        print(f"{goodbins}/{len(effsel)} good bins in selection function")

        assert (
            goodbins > 100
        ), f"Only {goodbins} valid bins in the Effective Selection Function"

        # Times by Area
        if "area" not in effsel_name:  # in older version I saved it that way
            print("x area...")
            for i in range(len(effsel)):
                for loc_i, loc in enumerate(self.rawsel._locations):
                    try:
                        effsel[i][loc_i] *= self.rawsel.area(loc)
                    except Exception as e:
                        # print(f"Encountered error, setting to 0, {e}")
                        effsel[i][loc_i] = effsel[i][loc_i] * 0

        # add that D**2 x delD factor to efffsel for fitting
        delD = self.distances[1] - self.distances[0]
        for i, f in enumerate(effsel):
            effsel[i] = effsel[i] * self.distances * self.distances * delD
        return effsel

    def load_effsel_allbins(self) -> Any:
        """Load in the effective selection function file for all bins"""
        # Load file
        effsel_name = os.path.join(
            PKG_ROOT, "data/selfuncs/effsel_allbins.npz"
        )
        effsel = np.load(effsel_name)["arr_0"]

        # Check to make sure most bins are valid
        goodbins = 0
        for i in range(len(effsel)):
            test = len(effsel[i][(effsel[i] != 0) & (np.isfinite(effsel[i]))])
            # print(test)
            if test >= 0.5 * (
                len(effsel[i].flatten())
            ):  # more than 90% of effsel is good
                goodbins += 1

        print(f"{goodbins}/{len(effsel)} good bins in selection function")

        assert (
            goodbins > 100
        ), f"Only {goodbins} valid bins in the Effective Selection Function"

        return effsel

    def calculate_coords(self) -> dict[str, Any]:
        """Caclulate the coordinates for the effective selection function"""
        # Coordinates for Effective Seleciton Function
        effsel_glons = np.array(
            [self.rawsel.glonGlat(x)[0][0] for x in self.rawsel._locations]
        )
        effsel_glats = np.array(
            [self.rawsel.glonGlat(x)[1][0] for x in self.rawsel._locations]
        )
        distbins = self.distances * 1000  # kpc to pc

        effsel_lons = []
        effsel_lats = []
        effsel_dist = []

        for loc_i in range(len(self.rawsel._locations)):
            for dist_i in range(len(distbins)):
                effsel_lons.append(effsel_glons[loc_i])
                effsel_lats.append(effsel_glats[loc_i])
                effsel_dist.append(distbins[dist_i])

        effsel_lons = np.array(effsel_lons)
        effsel_lats = np.array(effsel_lats)
        effsel_dist = np.array(effsel_dist)

        effsel_coords = coordinates.SkyCoord(
            l=effsel_lons * units.degree,
            b=effsel_lats * units.degree,
            distance=effsel_dist * units.pc,
            frame="galactic",
        )

        effsel_coords = effsel_coords.transform_to(coordinates.Galactocentric)

        effsel_xs = effsel_coords.x / 1000 / units.pc
        effsel_ys = effsel_coords.y / 1000 / units.pc
        effsel_zs = effsel_coords.z / 1000 / units.pc
        effsel_rs = np.sqrt((effsel_xs**2.0) + (effsel_ys**2.0))

        effsel_coords = {
            "x": effsel_xs,
            "y": effsel_ys,
            "z": effsel_zs,
            "r": effsel_rs,
            "rawsel_glon": effsel_glons,
            "rawsel_glat": effsel_glats,
            "effsel_glon": effsel_lons,
            "effsel_glat": effsel_lats,
        }
        return effsel_coords

    def calc_survey_volume(self) -> Any:
        """
        Calculate the physical volume sampled by the effective selection
        function (in kpc^3)
        """
        solid_angles = []
        for loc in self.rawsel._locations:
            try:
                solid_angles.append(self.rawsel.radius(loc))
            except Exception as e:
                solid_angles.append(np.nan)

        number_of_fields = len(solid_angles)
        max_distance = np.max(self.distances)

        # Using volume of cone = 1/3 pi r_base^2 h
        field_volumes = []
        for field_i in range(number_of_fields):
            solid_angle = units.arcsec.to(units.radian, solid_angles[field_i])
            r_base = max_distance * np.tan(solid_angle)
            volume_of_cone = (1.0 / 3.0) * np.pi * (r_base**2) * max_distance
            field_volumes.append(volume_of_cone)

        # Adding up as cylinders over all the distance slices
        # field_volumes = []
        # delD = self.distances[1] - self.distances[0]
        # for field_i in range(number_of_fields):
        #     solid_angle = units.arcsec.to(units.radian, solid_angles[field_i])
        #     field_volume = 0
        #     for distance_i in range(self.ndistmods):
        #         # middle value for r
        #         slice_r = self.distances[distance_i] - (delD / 2)
        #         r_cylind = slice_r * np.tan(solid_angle)
        #         cylind_volume = np.pi * delD * (r_cylind**2)
        #         field_volume += cylind_volume
        #     field_volumes.append(field_volume)

        return np.nansum(field_volumes)
        # 0.00128230579328566 using cone
        # # 0.0012822739632116864 using cylinders

    def calc_eff_survey_volume(self: Any, effsel_i: int) -> Any:
        """
        Calculate the effective survey volume (in kpc^3) for the selection
        function. The effective survey volume is the physical volume times the
        selection fraction integrated over distance.
        """
        solid_angles = []
        for loc in self.rawsel._locations:
            try:
                solid_angles.append(self.rawsel.radius(loc))
            except Exception as e:
                solid_angles.append(np.nan)

        number_of_fields = len(solid_angles)
        max_distance = np.max(self.distances)

        maap_effsel = self.effsel_noarea[effsel_i]

        # Adding up as cylinders over all the distance slices
        field_volumes = []  # total volume per pencil beam
        slice_volumes = []  # volume per slice of pencil beam
        delD = self.distances[1] - self.distances[0]
        for field_i in range(number_of_fields):
            solid_angle = units.arcsec.to(units.radian, solid_angles[field_i])
            field_volume = 0
            for distance_i in range(self.ndistmods):
                # middle value for r
                slice_r = self.distances[distance_i] - (delD / 2.0)
                r_cylind = slice_r * np.tan(solid_angle)
                cylind_volume = np.pi * delD * (r_cylind**2)
                field_volume += (
                    cylind_volume * maap_effsel[field_i][distance_i]
                )
                slice_volumes.append(
                    cylind_volume * maap_effsel[field_i][distance_i]
                )
            field_volumes.append(field_volume)

        return [np.nansum(field_volumes), slice_volumes]

    def plot_rawsel(self, ax: plt.Axes) -> None:
        """Plot the raw selection function"""

        # Get fractions for short cohort
        rawsel_colors = (
            self.rawsel._nspec_short / self.rawsel._nphot_short
        ).T[0] * 100

        idx = np.argsort(rawsel_colors)

        # Wrap latitude around 180 so it ranges -180 to 180
        # instead of 0 to 360
        lat = []
        for f in self.coordinates["rawsel_glon"][idx]:
            if f <= 180:
                lat.append(f)
            else:
                lat.append(f - 360)

        # Scatter Plot
        im = ax.scatter(
            np.deg2rad(lat),
            np.deg2rad(self.coordinates["rawsel_glat"][idx]),
            c=rawsel_colors[idx],
            vmin=0,
            vmax=100,
            cmap="viridis",
            s=50,
        )

        # Axes labels
        tick_values = np.deg2rad(np.arange(-150, 180, 30))
        ax.set_xticks(tick_values)
        ax.set_xticklabels(["" for i in tick_values])
        for lon in np.arange(-150, 180, 30):
            tx = ax.text(
                np.deg2rad(lon),
                np.deg2rad(-20),
                str(lon) + r"$\degree$",
                color="k",
                ha="center",
                va="top",
                fontsize=36,
            )
            tx.set_path_effects(
                [PathEffects.withStroke(linewidth=3, foreground="w")]
            )

        ydeg = np.arange(-75, 76, 15)
        ax.set_yticks(np.deg2rad(ydeg))
        ydeg_labels = [str(y) + r"$\degree$" for y in ydeg]
        ydeg_labels[-1] = ""
        ax.set_yticklabels(ydeg_labels)

        cbar = plt.colorbar(
            im, ax=ax, label="Selection Fraction (%)"
        )  # ,orientation='horizontal')
        ax.set_title("Raw Selection Function", fontsize=36)
        ax.grid()
        ax.scatter(
            [0],
            [0],
            marker="+",
            c="w",
            zorder=11,
            s=300,
            edgecolor="w",
            label="GC",
            lw=5,
            snap=False,
        )
        ax.scatter(
            [0],
            [0],
            marker="+",
            c="k",
            zorder=12,
            s=200,
            edgecolor="w",
            label="GC",
            lw=2,
            snap=False,
        )

        ax.set_xlabel("Galactic Longitude", fontsize=28)
        ax.set_ylabel("Galactic Latitude", fontsize=28)

    def plot_effsel(self, ax: plt.Axes) -> None:
        """Plot the effective selection function"""
        # Limit to within 5 degrees of MW midplane
        diskmap = np.abs(self.coordinates["effsel_glat"]) <= 5

        plot_xs = self.coordinates["x"][diskmap]
        plot_ys = self.coordinates["y"][diskmap]
        plot_cs = self.effsel_noarea[100].flatten()[diskmap]
        idx = np.argsort(plot_cs)
        im = ax.scatter(
            plot_xs[idx],
            plot_ys[idx],
            c=np.log10(plot_cs[idx]),
            vmin=-4,
            vmax=-1,
            s=100,
            marker=".",
            cmap="viridis",
        )

        plt.colorbar(im, ax=ax, label=r"log($\mathfrak{S}$)")
        ax.set_ylim(-20, 20)
        ax.set_xlim(-30, 10)
        for r in np.arange(0, 50, 5):
            radcircle = plt.Circle(
                (0, 0), r, facecolor="None", edgecolor="lightgray"
            )
            ax.add_patch(radcircle)

        # Set labels
        ax.set_xlabel("x (kpc)")
        ax.set_ylabel("y (kpc)")
        ax.set_aspect("equal")
        ax.set_title("Effective Selection Function", fontsize=36)


def resample_effsel(
    effsel_dict: dict[str, Any], new_r: Any, new_z: Any
) -> Any:
    """Resample an effsel to a different R,Z points"""
    current_effsel = effsel_dict["bin_effsel"]
    # effsel sampled at location of the data
    resampled_effsel = []
    for i in range(len(new_r)):
        point_distances = np.sqrt(
            (effsel_dict["bin_effsel_rs"] - new_r[i]) ** 2
            + (effsel_dict["bin_effsel_zs"] - new_z[i]) ** 2
        )
        closest_point = np.argmin(point_distances)
        resampled_effsel.append(current_effsel[closest_point])

    return np.array(resampled_effsel)


if __name__ == "__main__":
    set_env_variables()
    selfunc = SelectionFunction()
    # print volume (testing this)
    print(selfunc.calc_survey_volume())
