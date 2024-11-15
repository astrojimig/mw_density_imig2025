"""
mcmc_functions.py
Helper functions for performing the MCMC fitting of
the structural parameters of the Milky Way

Reference: J. Imig et al. 2024
"""

import emcee
import numpy as np
from mw_density.density_profiles import combined_exp_profile_nonsmoothed_linear
from mw_density.sample_selection import setup_maap_bins
from multiprocessing import Pool
from numpy.typing import NDArray
from typing import Any, Callable, Union
from astropy.io import fits


def set_param_limits() -> NDArray:
    """
    Set parameter limits for the MCMC search.
    It is not allowed to search for values outside of this range.
    """
    param_lims = np.array(
        [
            [-5.0, 5.0],  # limits for h_in
            [0.0, 5.0],  # limits for h_out
            [0.0, 2.5],  # limits for h_z
            [0.0, 30.0],  # limits for r_peak
            [0.0, 0.1],  # limits for a_flare
            [-10.0, 10.0],
        ]
    )  # limits for nu_0
    return param_lims


def mcmc_config() -> tuple[int, int, int]:
    """
    Define various parameters for the MCMC configuration.
    """
    # Set up some MCMC configuration
    # Number of Walkers
    nwalkers = 50
    # Burn-in period
    burn_in = 100
    # Number of iterations
    niter = 1000
    return nwalkers, burn_in, niter


def random_starting_guess(nwalkers: int) -> NDArray:
    """
    Choose a random starting guess from within the parameter limits
    """
    param_lims = set_param_limits()
    p0 = []
    for n in range(nwalkers):
        # choose random starting guesses within the limits
        h = np.random.choice(
            np.linspace(param_lims[0][0], param_lims[0][1], 100)
        )
        h2 = np.random.choice(
            np.linspace(param_lims[1][0], param_lims[1][1], 100)
        )
        hz = np.random.choice(
            np.linspace(param_lims[2][0], param_lims[2][1], 100)
        )
        rp = np.random.choice(
            np.linspace(param_lims[3][0], param_lims[3][1], 100)
        )
        rf = np.random.choice(
            np.linspace(param_lims[4][0], param_lims[4][1], 100)
        )
        norm = np.random.choice(
            np.linspace(param_lims[5][0], param_lims[5][1], 100)
        )
        p0.append([h, h2, hz, rp, rf, norm])
    return np.array(p0)


def check_bounds(theta: NDArray, r_sun: float = 8.122) -> bool:
    """
    Check it paramater array (theta) lies within the bounds
    defined in set_param_limits()
    """
    param_lims = set_param_limits()
    prior = True  # presumed in bounds until proven guilty
    # Check if all values are within the parameter limites
    for i in range(len(theta)):
        if (theta[i] < param_lims[i][0]) | (theta[i] > param_lims[i][1]):
            prior = False

    # extra prior for A_flare - h_z is not allowed to be negative.
    # todo: make theta a dictionary or something!!
    _, _, h_z0, _, a_flare, _ = theta
    if (h_z0 + a_flare * (0 - r_sun)) <= 0:
        prior = False

    return prior


def log_probability(
    theta: NDArray,
    densmodel: Callable,
    r_data: NDArray,
    z_data: NDArray,
    effsel_dat: NDArray,
    r_effsel: NDArray,
    z_effsel: NDArray,
) -> float:
    """
    Calculates the log likelihood that a given density model fits the data,
    this is the quanity we want to maximize.
    Inputs:
        theta: array: parameters for the dens_model
        densmodel: function: takes params, r, and z, and outputs density
        r_data: array: radial distribution of observed data
        z_data: array: z distribution of observed data
        effsel: (N_locations)x(N_DistanceBins) shape array:
            efffective selection function;
            percent of the intrinsic giant population obsevred by APOGEE
        r_effsel: array: the radial coordinates for the effsel grid
        z_efffsel: array: the z coordinates for the effsel grid.

    Outputs:
        log_likelihood: quanitity to minimize in MCMC
    """
    if not check_bounds(theta):
        return -np.inf  # Parameters are out of bounds

    log_dens = np.log(densmodel(theta, r_data, z_data))

    # expected number of stars in the survey given the model
    # (if density at sun is 1 pc-3):
    model_effsel = densmodel(theta, r_effsel, z_effsel)
    effective_volume = np.log(np.nansum(model_effsel * effsel_dat))
    if np.nansum(log_dens) == 0.0 or effective_volume == 0.0:
        return -np.inf

    if np.isfinite(np.nansum(log_dens)) & np.isfinite(effective_volume):
        # original version - no amplitude included
        log_likelihood = np.nansum(
            log_dens - np.ones(len(log_dens)) * effective_volume
        )
        # add amplitude
        # norm = 10.0**theta[-1]
        # log_likelihood = log_likelihood -
        # # continuednp.log(norm*np.nansum(model_effsel*effsel_dat))
        # add amplitude
        log_likelihood = log_likelihood - (1 + theta[-1]) * effective_volume

        if np.isnan(log_likelihood):
            return -np.inf
        else:
            return log_likelihood

    else:
        return -np.inf


def get_sample_data_mask(
    apogee_sample: fits.HDUList,
    metal_bin_i: int,
    age_bin_i: int,
    alpha_bin_val: str,
) -> NDArray:
    """
    Function to determine data mask for a given maap bin.

    Parameters:
    -----------
    metal_bin_i: int
        index of the target metal bin
    age_bin_i: int
        index of the target age bin
    alpha_bin: str
        value for alpha - 'LOW' or 'HIGH'

    Returns:
    --------
    data_binmask: list (Bool)
        Data mask for apogee_sample with the specified bin
    """
    data_binmask = apogee_sample["METAL_BIN_I"] == metal_bin_i
    data_binmask = data_binmask & (apogee_sample["AGE_BIN_I"] == age_bin_i)
    data_binmask = data_binmask & (apogee_sample["ALPHA_BIN"] == alpha_bin_val)
    return data_binmask


def perform_maap_density_fit(
    apogee_sample: fits.HDUList,
    effsel_dict: dict[str, Any],
    metal_bin_i: int,
    age_bin_i: int,
    alpha_bin_val: str,
    fname: str,
    nthreads: Union[None, int] = None,
) -> NDArray:
    """
    Function to perform the MCMC density fit for a give MAAP bin.

    Parameters:
    -----------
    apogee_sample: table
        Full APOGEE Sample
    metal_bin_i: int
        index of the target metal bin
    age_bin_i: int
        index of the target age bin
    alpha_bin: str
        value for alpha - 'LOW' or 'HIGH'
    fname: str
        filename to save results to
    initial_guess: array
        starting guess for the MCMC
    nthreads: int, optional
        Number of threads for parallelizing

    Returns:
    --------
    best_fit_params: array
        best fit parameter values from the MCMC

    """
    age_bins, mh_bins = setup_maap_bins()
    data_binmask = get_sample_data_mask(
        apogee_sample, metal_bin_i, age_bin_i, alpha_bin_val
    )
    n_stars = len(data_binmask[data_binmask])

    # MCMC settings
    nwalkers, burn_in, niter = mcmc_config()
    initial_guess = random_starting_guess(nwalkers)
    ndim = len(initial_guess[0])

    # Log some info
    maap_str = f"N = {n_stars} stars\n"
    maap_str += f"[M/H] = {round(mh_bins['center'][metal_bin_i],2)}, "
    maap_str += f"age = {round(age_bins['center'][age_bin_i],2)}, "
    maap_str += f"alpha = {alpha_bin_val}"
    print(maap_str)
    print("=" * 50)

    if n_stars < 100:
        print(
            f"n_stars = {n_stars} is less than minimum of 100. Skipping bin."
        )
    else:
        # Set up Selection Function for sample
        bin_effsel_rs = effsel_dict["bin_effsel_rs"]
        bin_effsel_zs = effsel_dict["bin_effsel_zs"]
        bin_effsel = effsel_dict["bin_effsel"]

        # Set up data for Sample:
        bin_r_data = apogee_sample["GALACTIC_R"][data_binmask]
        bin_z_data = apogee_sample["GALACTIC_Z"][data_binmask]

        # arguments for lnprob
        lnprob_args = [
            combined_exp_profile_nonsmoothed_linear,
            bin_r_data,
            bin_z_data,
            bin_effsel,
            bin_effsel_rs,
            bin_effsel_zs,
        ]

        if nthreads:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_probability,
                args=lnprob_args,
                pool=Pool(nthreads),
            )
        else:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability, args=lnprob_args
            )

        # Run the MCMC
        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(initial_guess, burn_in, progress=True)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

        # Save it out
        np.savez(
            fname,
            chain=sampler.flatchain,
            flatlnprobability=sampler.flatlnprobability,
        )

        best_fit_params = sampler.flatchain[
            np.argmax(sampler.flatlnprobability)
        ]
        print(best_fit_params)
        return best_fit_params
