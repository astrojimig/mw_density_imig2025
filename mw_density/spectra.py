"""
spectra.py
Code and helper functions for working with spectra, including
the MaStar SSP spectra and integrated Milky Way Analog Spectra

Reference: J. Imig et al. 2025
"""

import os
import astropy.io.fits as fits
import numpy as np
from numpy.typing import NDArray
from typing import Any

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================================
# Class for MaStar SSPs
# ==================================


class MaStarSSPs:
    """Class for working with the MaStar SSP Spectra"""

    def __init__(self) -> None:
        """Initate class instance and load in file"""
        print("Loading in MaStar SSP Spectra")
        print("=" * 50)
        self.filename = "data/ssps/MaStar_SSP_v0.2.fits"
        self.filepath = os.path.join(PKG_ROOT, self.filename)
        mastar_ssps = fits.open(self.filepath)
        self.ages = mastar_ssps[1].data.T[0][0][0]  # in Gyr
        self.fehs = mastar_ssps[1].data.T[1].T[0].T[0]
        self.imf_slopes = mastar_ssps[1].data.T[2].T[0][0]
        self.wls = mastar_ssps[2].data[0]
        self.spectra = mastar_ssps[4].data
        print("Complete")
        print("=" * 50)

    def get_ssp(self, target_age: float, target_feh: float) -> NDArray:
        """
        Retrieves an SSP for given age and metalicity,
        interpolating the grid if needed
        """

        # SSP grid only goes to 0.35
        if target_feh >= np.max(self.fehs):
            feh_i_2 = len(self.fehs) - 1
        else:
            feh_i_2 = np.where(self.fehs >= target_feh)[0][0]
        feh_i_1 = feh_i_2 - 1
        feh1 = self.fehs[feh_i_1]
        feh2 = self.fehs[feh_i_2]
        age_i_2 = np.where(self.ages >= target_age)[0][0]
        age_i_1 = age_i_2 - 1
        age1 = self.ages[age_i_1]
        age2 = self.ages[age_i_2]
        slope_i = 2  # Kroupa IMF

        spec1a = self.spectra[age_i_1][feh_i_1][slope_i]
        spec1b = self.spectra[age_i_2][feh_i_1][slope_i]

        spec2a = self.spectra[age_i_1][feh_i_2][slope_i]
        spec2b = self.spectra[age_i_2][feh_i_2][slope_i]

        target_spec = []

        for i in range(len(self.wls)):
            # Interpolate in feh first, and then in age
            p1 = np.interp(target_feh, [feh1, feh2], [spec1a[i], spec2a[i]])
            p2 = np.interp(target_feh, [feh1, feh2], [spec1b[i], spec2b[i]])
            target_spec.append(np.interp(target_age, [age1, age2], [p1, p2]))

        return np.array(target_spec)

    def light_to_mass(
        self, mh_bins: list[float], age_bins: list[float]
    ) -> NDArray:
        """
        Calculates the integrated flux per stellar population mass.
        """
        ssp_flux = np.zeros((len(mh_bins), len(age_bins)))
        for i_a, age in enumerate(age_bins):
            # Loop over metalicity bins
            for i_f, feh in enumerate(mh_bins):
                if feh <= np.max(self.fehs):
                    ssp_flux[i_f, i_a] = np.nansum(self.get_ssp(age, feh))
                else:
                    ssp_flux[i_f, i_a] = 0
            # extrapolate to get missing top row
            # The grid does not go metal rich enough
            pf = np.polyfit(mh_bins[:-1], ssp_flux[:-1, i_a], 3)
            x1 = mh_bins[-1]
            y1 = x1**3 * pf[0] + x1**2 * pf[1] + x1 * pf[2] + pf[3]
            ssp_flux[-1][i_a] = y1

        return ssp_flux

    def ssp_grid(self, mh_bins: list[float], age_bins: list[float]) -> NDArray:
        """
        Makes a grid of SSPs for use in stuff.
        """
        ssp_grid = np.zeros((len(mh_bins), len(age_bins), len(self.wls)))
        for i_a, age in enumerate(age_bins):
            for i_f, feh in enumerate(mh_bins):
                ssp_grid[i_a, i_f] = self.get_ssp(age, feh)
        return ssp_grid


# ==================================
# Generic Spectra functions
# ==================================


def sdss_filters() -> dict[str, Any]:
    """
    Load in the SDSS filter curves
    """
    filepath = "data/sloan_filters/filter_curves.fits"
    with fits.open(filepath) as filter_curves:
        filters = {}
        for i, s in enumerate(["u", "g", "r", "i", "z"]):
            filter_dict = {
                "name": s,
                "wavelength": filter_curves[i + 1].data["wavelength"],
                "resbig": filter_curves[i + 1].data["resbig"],
            }
            filters[f"{s}"] = filter_dict
    return filters


def measure_gr_color(spec_wls: NDArray, spec_flux: NDArray) -> float:
    """Measures the colors from a spectrum"""

    # Load in the SDSS filter profiles
    filters = sdss_filters()

    # Resample to target wavelength
    g_spec = np.interp(filters["g"]["wavelength"], spec_wls, spec_flux)
    r_spec = np.interp(filters["r"]["wavelength"], spec_wls, spec_flux)

    # Sum over flux
    g_flux = np.nansum(
        g_spec * filters["g"]["resbig"] * filters["g"]["wavelength"]
    )
    r_flux = np.nansum(
        r_spec * filters["r"]["resbig"] * filters["r"]["wavelength"]
    )

    # Convert to magnitude
    # Reference values from AB mag system
    # https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
    # https://www.sdss.org/dr12/algorithms/ugrizvegasun/
    # 3631 Jy: this is constant in NU not wl
    # f_nu = 3631 * np.ones(len(spec_wls))
    flux_ab = (
        3631.0 * np.ones(len(spec_wls)) * ((3.0e8) / (spec_wls * 1.0e-10))
    )
    ref_mags = {}
    # for i, s in enumerate(["u", "g", "r", "i", "z"]):
    for i, s in enumerate(["g", "r"]):
        filter_curve = filters[s]["resbig"]
        filter_spec = np.interp(filters[s]["wavelength"], spec_wls, flux_ab)
        ref_mags[s] = -2.5 * np.log10(
            np.nansum(filter_curve * filter_spec * filters[s]["wavelength"])
        )

    g_mag = -2.5 * np.log10(g_flux) - ref_mags["g"]
    r_mag = -2.5 * np.log10(r_flux) - ref_mags["r"]

    return g_mag - r_mag
