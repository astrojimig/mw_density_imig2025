"""
sample_selection.py
Definitions for MAAP stellar population bins, and
various functions for selecting the RGB sample from allStar

Reference: J. Imig et al. 2024
"""

import os
import sys
import datetime
import numpy as np
from astropy.table import Column
from astropy.io import fits
from astropy import units, coordinates, table
from typing import Any, Callable, Union
from numpy.typing import NDArray

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ========================
# Set file paths and environment variables
# ========================
def set_env_variables() -> None:
    """
    Set environment variables
    """
    sys.path.insert(1, "/Users/jimig/research/apogee/")

    results_vers = "dr17"
    os.environ["RESULTS_VERS"] = results_vers

    # Local SAS mirror path
    sas_mirror_path = "/Users/jimig/research/sas_mirror/"
    os.environ["SDSS_LOCAL_SAS_MIRROR"] = sas_mirror_path

    # Directory containing dust maps
    dust_dir_path = "/Users/jimig/research/mwdust/dustmaps/"
    os.environ["DUST_DIR"] = dust_dir_path


def get_allstar_filepath() -> str:
    """
    Define allstar filepath
    """
    # allStar filepath - used fo stellar Parameters
    allstar_filepath = (
        "/Users/jimig/sdss/apogee/allStar-dr17-synspec_rev1.fits"
    )
    return allstar_filepath


def get_distmass_filepath() -> str:
    """
    Define distmass filepath
    """
    # Distmass Catalog - Age and Distance Estimates
    distmass_filepath = (
        "/Users/jimig/research/apogee-vacs/APOGEE_DistMass-DR17_v1.6.1.fits"
    )
    return distmass_filepath


# ========================
# STELLAR POPULATION BINS:
# age and [M/H] bin definition for MAAPs
# ========================

# type alias for maap array
bin_type = dict[str, NDArray]


def setup_maap_bins() -> tuple[bin_type, bin_type]:
    """
    Define limits for MAAP bins
    """
    # Metallicity bins
    delta_mh = 0.1
    # feh_bins_center = np.arange(-0.65,0.46,delta_mh)
    mh_bins_center = np.arange(-0.95, 0.46, delta_mh)

    mh_bins_min = mh_bins_center - (delta_mh / 2.0)
    mh_bins_max = mh_bins_center + (delta_mh / 2.0)

    # Age bins
    delta_age = 0.1  # (in log10 space)
    age_bins_center = 10.0 ** np.arange(9.05, 10.16, delta_age)
    age_bins_min = 10.0 ** (np.log10(age_bins_center) - delta_age / 2)
    age_bins_max = 10.0 ** (np.log10(age_bins_center) + delta_age / 2)

    age_bins_min = age_bins_min / 1e9
    age_bins_max = age_bins_max / 1e9
    age_bins_center = age_bins_center / 1e9

    age_bins = {
        "center": age_bins_center,
        "min": age_bins_min,
        "max": age_bins_max,
    }

    mh_bins = {
        "center": mh_bins_center,
        "min": mh_bins_min,
        "max": mh_bins_max,
    }

    return age_bins, mh_bins


def define_rsun() -> float:
    """
    Provide a definition for Rsun
    """
    r_sun = 8.122
    return r_sun
    # TODO: Use this is in coordinate definition!! (currenltly 8.122)


# ========================
# DATA MASK DEFINITION
# ========================
def get_data_mask(allstar: fits.HDUList, distmass: fits.HDUList) -> NDArray:
    """
    Generate mask for data
    """
    age_bins, mh_bins = setup_maap_bins()

    # Allstar Masks
    # Red Giants
    allstar_mask = (allstar["LOGG"] <= 3.5) & (allstar["LOGG"] >= 0.8)
    # Main Survey Sample
    allstar_mask = allstar_mask & (allstar["EXTRATARG"] == 0)
    # Good parameters
    allstar_mask = (
        allstar_mask & (allstar["ALPHA_M"] >= -1.0) & (allstar["TEFF"] <= 5500)
    )
    allstar_mask = allstar_mask & (
        (allstar["ASPCAPFLAG"] & (2**23)) == 0
    )  # STAR_BAD
    # Within metal bins
    allstar_mask = allstar_mask & (allstar["M_H"] >= np.min(mh_bins["min"]))
    allstar_mask = allstar_mask & (allstar["M_H"] <= np.max(mh_bins["max"]))
    # Falls in one of low or high alpha zones
    low_alpha_mask, high_alpha_mask = get_alpha_masks(allstar)
    allstar_mask = allstar_mask & (low_alpha_mask | high_alpha_mask)

    distmassmask = (distmass["TEFF"] < 4500) & (distmass["LOGG"] > 3)
    distmassmask = (
        distmassmask | (distmass["LOGG"] > 4.9) | (distmass["TEFF"] > 6800)
    )
    distmassmask = distmassmask | (
        (distmass["TEFF"] > 5500) & (distmass["LOGG"] < 3.1)
    )
    distmassmask = distmassmask | (distmass["LOGG"] < 0.8)
    distmassmask = distmassmask | (
        distmass["LOGG"] < (0.000906 * distmass["TEFF"] - 2.625)
    )
    distmassmask = distmassmask | (
        (distmass["LOGG"] < 2)
        & (distmass["TEFF"] > 4800)
        & (distmass["TEFF"] < 5500)
    )
    distmassmask = distmassmask | (
        (distmass["LOGG"] > (0.002 * distmass["TEFF"] - 5.8))
        & (distmass["LOGG"] < 4.1)
    )
    # distmassmask = distmassmask | ((distmass['M_H']<-0.7))
    distmassmask = np.invert(np.array(distmassmask))
    distmassmask = distmassmask & (distmass["WEIGHTED_DIST"] <= 2.5e5)
    # Within age bins
    distmassmask = distmassmask & (
        (distmass["AGE_UNCOR_SS"] / 1e9) >= np.min(age_bins["min"])
    )
    distmassmask = distmassmask & (
        (distmass["AGE_UNCOR_SS"] / 1e9) <= np.max(age_bins["max"])
    )
    distmass_mask = distmassmask
    # Full sample
    data_mask = allstar_mask & distmass_mask
    return data_mask


# ========================
# ALPHA GROUP DEFINITION
# ========================
def alpha_line_imig2023(
    m_h: Union[float, list, NDArray],
) -> Union[float, list, NDArray]:
    """
    Define line for splitting high and low alpha groups
    from Imig et al. 2023 (eq 1)
    https://ui.adsabs.harvard.edu/abs/arXiv:2307.13887

    Parameters
    ----------
    m_h : list
        of [M/H] values for which to calculate the alpha value

    Returns
    -------
    alpha_line: list
        values for [Alpha/M] which determine the population split

    """
    m_h_vals = [-1.0, -0.5, 0.0, 0.5]
    aplha_vals = [0.12 - (0.13 * -1), 0.12 - (0.13 * -0.5), 0.12, 0.12]
    alpha_line = np.interp(m_h, m_h_vals, aplha_vals)
    return alpha_line


def alpha_line_patil2023(
    m_h: Union[float, NDArray],
) -> Union[float, NDArray]:
    """
    Define line for splitting high and low alpha groups
    from Patil et al. 2023 (eq 24)
    https://ui.adsabs.harvard.edu/abs/arXiv:2306.09319

    Parameters
    ----------
    m_h : list
        of [M/H] values for which to calculate the alpha value

    Returns
    -------
    alpha_line: list
        values for [Alpha/M] which determine the population split

    """
    alpha_line = (
        0.1754 * (m_h**3) + 0.1119 * (m_h**2) - 0.1253 * (m_h) + 0.1353
    )
    # extra -0.05 for [Fe/H] to [M/H] calibration
    return alpha_line - 0.05


def get_alpha_masks(
    allstar: fits.HDUList, line_func: Callable = alpha_line_patil2023
) -> tuple[Any, Any]:
    """
    Generate high- and low-alpha samples based on line
    Parameters
    ----------
    allstar : astropy Table of allStar file
    line_func : function for defining the line on which to split
    (default Patil et al. 2023)

    Returns
    -------
    low_alpha_mask, high_alpha_mask : list, bools
        Mask for determining high- and low-alpha samples

    """
    test_alps = line_func(allstar["M_H"])
    high_alpha_mask = allstar["ALPHA_M"] >= test_alps  # + 0.025
    low_alpha_mask = allstar["ALPHA_M"] <= test_alps  # - 0.025
    # Impose metallicity limit on low-alpha stars (for distmass ages)
    low_alpha_mask = low_alpha_mask & (allstar["M_H"] >= -0.7)
    return low_alpha_mask, high_alpha_mask


# ===========================
# Coordinates and Parameters
# ===========================


def calc_coordinates(
    allstar: fits.HDUList, distmass: fits.HDUList, data_mask: NDArray
) -> dict[str, Any]:
    """
    Calculate cartesian galactic X,Y,Z from RA, DEC and distance
    """

    data_coords = coordinates.SkyCoord(
        ra=allstar["RA"][data_mask] * units.degree,
        dec=allstar["DEC"][data_mask] * units.degree,
        distance=distmass["WEIGHTED_DIST"][data_mask] * units.pc,
        frame="icrs",
        radial_velocity=allstar["VHELIO_AVG"][data_mask] * units.km / units.s,
        pm_dec=allstar["GAIAEDR3_PMDEC"][data_mask] * units.mas / units.yr,
        pm_ra_cosdec=allstar["GAIAEDR3_PMRA"][data_mask]
        * units.mas
        / units.yr,
    )

    data_coords = data_coords.transform_to(coordinates.Galactocentric)

    coord_dict = {
        "x": data_coords.x,
        "y": data_coords.y,
        "z": data_coords.z,
        "r": np.sqrt(((data_coords.x) ** 2.0) + ((data_coords.y) ** 2.0)),
    }

    return coord_dict


def distmod_bins() -> tuple[int, Any, Any, Any]:
    """
    Define distance mod bins for selection function
    """
    # Distance bins
    ndistmods = 100  # number of distance bins
    distances = np.linspace(0.01, 25.0, ndistmods)  # Distance in kpc
    # Convert from distance in kpc to distance modulus
    distmods = 5.0 * np.log10(distances * 1000) - 5.0
    minmax_distmods = [np.min(distmods), np.max(distmods)]
    return ndistmods, distances, distmods, minmax_distmods


def assume_high_alpha_ages(
    allstar: fits.HDUList, distmass: fits.HDUList
) -> Any:
    """
    Assume ages for high-alpha, metal-poor stars
    """
    age_bins, mh_bins = setup_maap_bins()
    _, high_alpha_mask = get_alpha_masks(allstar)
    orig_ages = np.log10(distmass["AGE_UNCOR_SS"])  # work in log ages

    orig_age_bin = []
    for age in orig_ages:
        orig_age_bin.append(
            (np.abs(np.log10(age_bins["center"] * 1e9) - age).argmin())
        )
    orig_age_bin = np.array(orig_age_bin)

    metal_bin = []
    for m_h in allstar["M_H"]:
        metal_bin.append((np.abs(mh_bins["center"] - m_h)).argmin())
    metal_bin = np.array(metal_bin)

    # Age distribution to match
    target_mask = (
        (high_alpha_mask) & (allstar["M_H"] >= -0.6) & (allstar["M_H"] <= -0.4)
    )
    goal_fraction = []
    for i_a, age in enumerate(age_bins["center"]):
        agebin_mask = (target_mask) & (orig_age_bin == i_a)
        frac = len(orig_ages[agebin_mask]) / len(orig_ages[target_mask])
        goal_fraction.append(frac)

    # target percentile in each age bin
    target_percentiles = np.cumsum(np.array(goal_fraction)) * 100
    assumed_ages = np.copy(orig_ages)
    # Calculate new assumed age
    for star_i, star_age in enumerate(assumed_ages):
        # If star is in the assumed ages zone
        if (high_alpha_mask[star_i]) & (allstar["M_H"][star_i] <= (-0.7)):
            pop_mask = (high_alpha_mask) & (metal_bin == metal_bin[star_i])
            pop_ages = orig_ages[pop_mask]
            age_percentiles = np.percentile(pop_ages, target_percentiles)
            age_percentiles[-1] = (
                np.max(pop_ages) + 0.1
            )  # large one just for edge cases
            percentile_index = np.where(star_age <= age_percentiles)[0][0]
            assumed_ages[star_i] = np.log10(age_bins["center"] * 1e9)[
                percentile_index
            ]

    return (10.0 ** (assumed_ages)) / 1.0e9


# ========================
# GENERATE SAMPLE
# ========================


def generate_mwd_sample(save_filename: str = "mwd_sample.fits") -> None:
    """
    Generate the data sample using the bins and masks defined above
    """
    allstar = fits.open(get_allstar_filepath())[1].data
    distmass = fits.open(get_distmass_filepath())[1].data
    data_mask = get_data_mask(allstar, distmass)
    age_bins, mh_bins = setup_maap_bins()
    # Caclulate alpha masks
    low_alpha_mask, high_alpha_mask = get_alpha_masks(allstar[data_mask])
    # Calculate Coordinates
    data_coords = calc_coordinates(allstar, distmass, data_mask)

    # Define Population Bins
    alpha_group = []
    for star_i, low_alpha in enumerate(low_alpha_mask):
        if low_alpha:
            alpha_group.append("LOW")
        elif high_alpha_mask[star_i]:
            alpha_group.append("HIGH")
        # else:
        #    alpha_group.append("NONE")
        # None should be NONE!

    metal_bin = []
    for m_h in allstar["M_H"][data_mask]:
        metal_bin.append((np.abs(mh_bins["center"] - m_h)).argmin())

    # For high-alpha, low-metallicity stars, need to assume an age distribution
    assumed_ages = assume_high_alpha_ages(
        allstar[data_mask], distmass[data_mask]
    )

    # Calculate age bin using new ages
    age_bin = []
    for age in assumed_ages:
        # Must be in log space
        age_bin.append(
            (
                np.abs(
                    np.log10(age_bins["center"] * 1e9) - np.log10(age * 1e9)
                )
            ).argmin()
        )

    # Save out file
    fitstable = table.Table()
    for col in [
        # APOGEE ID
        Column(
            allstar["APOGEE_ID"][data_mask],
            name="APOGEE_ID",
            description="APOGEE ID from allStar",
        ),
        # APOGEE COORDINATES
        Column(
            allstar["RA"][data_mask],
            name="RA",
            description="RA from allStar",
            unit="deg",
        ),
        Column(
            allstar["DEC"][data_mask],
            name="DEC",
            description="declination from allStar",
            unit="deg",
        ),
        # STELLAR PARAMETERS
        Column(
            allstar["M_H"][data_mask],
            name="M_H",
            description="metallicity from allStar",
        ),  # , unit='dex'),
        Column(
            allstar["ALPHA_M"][data_mask],
            name="ALPHA_M",
            description="alpha abundance from allStar",
        ),  # , unit='dex'),
        Column(
            allstar["LOGG"][data_mask],
            name="LOGG",
            description="LOGG from allStar",
        ),  # , unit='dex'),
        Column(
            allstar["TEFF"][data_mask],
            name="TEFF",
            description="TEFF from allStar",
            unit="K",
        ),
        # AGES
        Column(
            assumed_ages,
            name="AGE",
            description="stellar age from distmass + high-alpha assumptions",
            unit="Gyr",
        ),
        Column(
            np.log10(assumed_ages * 1.0e9),
            name="LOG_AGE",
            description="log stellar age",
        ),  # , unit='log10(yr)'),
        Column(
            distmass["AGE_UNCOR_SS"][data_mask] / 1.0e9,
            name="DISTMASS_AGE",
            description="stellar age from distmass",
            unit="Gyr",
        ),
        Column(
            distmass["WEIGHTED_DIST"][data_mask] / 1000.0,
            name="DISTMASS_DIST",
            description="weighted distances from distmass",
            unit="kpc",
        ),
        # GALACTOCENTRIC COORDINATES
        Column(
            data_coords["x"],
            name="GALACTIC_X",
            description="Galactocentric Coordinates",
            unit="kpc",
        ),
        Column(
            data_coords["y"],
            name="GALACTIC_Y",
            description="Galactocentric Coordinates",
            unit="kpc",
        ),
        Column(
            data_coords["z"],
            name="GALACTIC_Z",
            description="Galactocentric Coordinates",
            unit="kpc",
        ),
        Column(
            data_coords["r"],
            name="GALACTIC_R",
            description="Galactocentric radius",
            unit="kpc",
        ),
        # MAAP DEFINITIONS
        Column(alpha_group, name="ALPHA_BIN", description="MAAP alpha bin"),
        Column(
            mh_bins["center"][metal_bin],
            name="METAL_BIN",
            description="MAAP metallicity bin",
        ),
        Column(
            metal_bin,
            name="METAL_BIN_I",
            description="MAAP metallicity bin index",
        ),
        Column(
            age_bins["center"][age_bin],
            name="AGE_BIN",
            description="MAAP age bin",
        ),
        Column(age_bin, name="AGE_BIN_I", description="MAAP age bin index"),
    ]:
        fitstable.add_column(col)

    fitstable.write(save_filename, format="fits", overwrite=True)
    # Update header (to do: fix header - this corrupts it??)
    # write_file_header(save_filename)


def write_file_header(save_filename: str) -> None:
    """
    Populates the file header with various metadata
    """
    with fits.open(f"{save_filename}", mode="update") as hdul:
        hdr = hdul[0].header
        hdr["DATE"] = (
            f"{datetime.datetime.now().__str__()}",
            "Time of File Creation",
        )
        hdr["AUTHOR"] = ("Julie Imig (STScI)", "Primary Author")
        hdr["REFERENC"] = ("Imig et al. 2025", "Primary Reference")
        if "apogee_sample" in save_filename:
            hdr["DESCRIP"] = (
                "APOGEE RGB sample data used in Imig et al 2024",
                "description",
            )
        elif "density_params" in save_filename:
            hdr["DESCRIP"] = (
                "Milky Way Density Measurements from Imig et al 2024",
                "description",
            )

        hdr["ALLSTAR"] = (
            f"{get_allstar_filepath().split('/')[-1]}",
            "allStar file version used",
        )
        hdr["DISTMASS"] = (
            f"{get_distmass_filepath().split('/')[-1]}",
            "distmass file version used",
        )
        hdul.flush()  # changes are written back to original file


if __name__ == "__main__":
    print("Generating APOGEE Sample for density fitting...")
    SAVE_NAME = os.path.join(PKG_ROOT, "data", "apogee_sample.fits")
    generate_mwd_sample(SAVE_NAME)
    print(f"Saved to {SAVE_NAME}.")
