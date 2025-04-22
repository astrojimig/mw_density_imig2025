"""
isochrones.py
Code and helper functions for working with the isochrone files

Reference: J. Imig et al. 2025
"""

import os
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from mw_density.sample_selection import setup_maap_bins
from tqdm import tqdm

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Isochrones:
    """Class for working with the isocrhone files"""

    def __init__(
        self, isochrone_filename: str = "data/isochrones/parsec.dat"
    ) -> None:
        print(f"Loading in Isochrones from '{isochrone_filename}'")
        print("=" * 50)

        file_path = os.path.join(PKG_ROOT, isochrone_filename)
        self.JHK_iso = pd.read_csv(file_path, sep=r"\s+", comment="#")
        self.delta_mass = self.add_delta_mass_column()
        age_bins, mh_bins = setup_maap_bins()
        self.age_bins = age_bins
        self.mh_bins = mh_bins
        self.mass_ratio = self.calc_mass_ratio()
        print("Complete.")
        print("=" * 50)

    def add_delta_mass_column(self) -> NDArray:
        """
        Adds a deltaM column to the isochrones

        Delta M is calcualted from the int_IMF column:
        'Differences between 2 values of int_IMF give the absolute number of
        stars occupying that isochrone section per unit mass of stellar
        population initially born, as expected for the selected IMF.'

        https://stev.oapd.inaf.it/cmd_3.8/help.html

        """
        delta_mass = []
        print("Calculating IMF values from isochrone points")
        for i in tqdm(range(len(self.JHK_iso["MH"]))):
            if i > 0:
                same_mh = (
                    np.array(self.JHK_iso["MH"])[i - 1]
                    == np.array(self.JHK_iso["MH"])[i]
                )
                same_age = (
                    np.array(self.JHK_iso["logAge"])[i - 1]
                    == np.array(self.JHK_iso["logAge"])[i]
                )
            else:  # first row
                same_mh = False
                same_age = False

            if same_age & same_mh:
                delta_mass.append(
                    np.array(self.JHK_iso["int_IMF"])[i]
                    - np.array(self.JHK_iso["int_IMF"])[i - 1]
                )
            else:
                delta_mass.append(
                    np.array(self.JHK_iso["int_IMF"])[i + 1]
                    - np.array(self.JHK_iso["int_IMF"])[i]
                )

        return np.array(delta_mass)
        # TO DO int_IMF[-1] - int_IMF[0] is really all that I should need.
        # Just use that?

    def isochrone_mask_for_maap(
        self,
        i_a: int,
        i_m: int,
        return_mask: bool = False,
    ) -> list[bool]:
        """
        Mask Isochrone points for desired MAAPP bin, and logg cuts
        """
        # Cut to metallicity and age limits
        m = self.JHK_iso["MH"] >= self.mh_bins["min"][i_m]
        m = m & (self.JHK_iso["MH"] <= self.mh_bins["max"][i_m])
        min_age = np.log10(self.age_bins["min"][i_a] * 1.0e9)
        max_age = np.log10(self.age_bins["max"][i_a] * 1.0e9)
        m = m & (self.JHK_iso["logAge"] >= min_age)
        m = m & (self.JHK_iso["logAge"] <= max_age)

        # Cut off pre-main sequence and post-AGB stars
        # https://stev.oapd.inaf.it/cmd_3.8/help.html
        m = m & (self.JHK_iso["label"] < 9) & (self.JHK_iso["label"] > 0)

        if return_mask:
            return m
        else:
            return self.JHK_iso[m]

    def calc_mass_ratio(self) -> NDArray:
        """
        Calculate the number ratio of giant stars to
        population mass conversion ratio
        """
        mass_ratio = np.zeros(
            (len(self.mh_bins["center"]), len(self.age_bins["center"]))
        )

        for i_m in range(len(self.mh_bins["center"])):
            for i_a in range(len(self.age_bins["center"])):
                maap_mask = self.isochrone_mask_for_maap(
                    i_a, i_m, return_mask=True
                )
                # min and max from apogee sample
                logg_min_lim = 0.8000012
                logg_max_lim = 3.4999957
                logg_mask = self.JHK_iso["logg"] <= logg_max_lim
                logg_mask = logg_mask & (self.JHK_iso["logg"] >= logg_min_lim)
                logg_mask = logg_mask & maap_mask

                # From sum of consecutive differences
                # mass_in_giants = np.nansum(self.delta_mass[logg_mask])
                # tot_mass = np.nansum(self.delta_mass[maap_mask])
                #  From difference between last and first
                int_imf = np.array(self.JHK_iso["int_IMF"])
                mass_in_giants = int_imf[logg_mask][-1] - int_imf[logg_mask][0]
                tot_mass = int_imf[maap_mask][-1] - int_imf[maap_mask][0]
                # This is the same as max - min!
                # mass_in_giants = np.max(int_imf[logg_mask]) - np.min(
                #     int_imf[logg_mask]
                # )
                # tot_mass = np.max(int_imf[maap_mask]) - np.min(
                #     int_imf[maap_mask]
                # )

                # Add to array
                mass_ratio[i_m][i_a] = mass_in_giants / tot_mass

        return 1.0 / mass_ratio
