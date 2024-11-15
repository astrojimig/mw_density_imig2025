"""
selfunc.py
Code and helper functions for calculating the APOGEE
raw and effective selection functions

Reference: J. Imig et al. 2024
"""

import dill as pickle
import numpy as np
from mw_density.sample_selection import distmod_bins
from typing import Any
import os
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SelectionFunction:
    def __init__(self) -> None:
        # Load in files
        # Raw Selection Function
        self.rawsel = self.load_raw_selfunc()
        # Effective Selection Function
        self.effsel = self.load_effsel_allbins()
        # Set distmod bins from sample selection
        self.distmod_bins = distmod_bins()
        self.ndistmods, self.distances, self.distmods, self.minmax_distmods = (
            self.distmod_bins
        )
        # Calculate coordinates for effsel
        self.coordinates = self.effsel_coords()

    def load_raw_selfunc(self) -> Any:
        """Load in the raw selection function file"""
        rawsel_path = os.path.join(
            PKG_ROOT, "data/selfuncs/apogeeCombinedSF.dat"
        )
        with open(rawsel_path, "rb") as f:
            rawsel = pickle.load(f)
        return rawsel

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
                        print(f"Encountered error, setting to 0, {e}")
                        effsel[i][loc_i] = effsel[i][loc_i] * 0

        # add that D**2 x delD factor to efffsel for fitting
        delD = self.distances[1] - self.distances[0]
        for i, f in enumerate(effsel):
            effsel[i] = effsel[i] * self.distances * self.distances * delD
        return effsel

    def effsel_coords(self) -> None:
        """Caclulate the coordinates for the effective selection function"""
        ...

    def resample_effsel(self) -> None:
        """Resample an effsel to a different R,Z points"""
        ...
