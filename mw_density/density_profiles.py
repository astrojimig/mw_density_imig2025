"""
density_profiles.py
Density Profiles and equations for the MW

Reference: J. Imig et al. 2024
"""

import numpy as np
from numpy.typing import NDArray


# ====================
# Density Models
# ====================


def radial_broken_exp_smoothed(
    h_r_in: float,
    h_r_out: float,
    r_peak: float,
    r_data: NDArray,
    smoothed_radius: float = 1.0,
    r_sun: float = 8.122,
) -> NDArray:
    """Smoothed Broken Exponential for MW modeling.
    Inputs:
        h_r_in: float: 1/inner exponetial parameter
        h_r_out: float: 1/outer exponetial parameter
        r_peak: float: break radius (kpc)
        r_data: array: array of radii values to evaluate at (kpc)
        smoothed_radius: optional, float: radius for smoothing
            around r_peak (default 1.0 kpc)
    Outputs:
        radial profile for array r
    """
    r0 = r_sun
    r_data = np.abs(r_data)
    sort_index = np.argsort(r_data)
    reverse_sort = np.argsort(sort_index)
    r_data = r_data[sort_index]

    rp_in = (-1.0 * h_r_in) * (r_peak - r0)
    rp_out = (-1.0 * h_r_out) * (r_peak - r0)
    norm_constant = rp_in - rp_out

    inmask = r_data <= (r_peak - smoothed_radius)
    outmask = r_data >= (r_peak + smoothed_radius)
    midmask = np.invert(np.logical_or(inmask, outmask))

    inprof = (-1.0 * h_r_in) * (r_data[inmask] - r0) - norm_constant
    outprof = (-1.0 * h_r_out) * (r_data[outmask] - r0)

    if len(midmask[midmask]) > 0:
        midrs = np.linspace(
            r_peak - smoothed_radius, r_peak + smoothed_radius, 100
        )
        midprof = np.zeros(len(midrs))
        for i, r in enumerate(midrs):
            test_rs = np.linspace(
                r - smoothed_radius, r + smoothed_radius, 100
            )
            tr_inmask = test_rs < r_peak
            tr_outmask = test_rs >= r_peak
            x_in = (-1.0 * h_r_in) * (test_rs[tr_inmask] - r0) - norm_constant
            x_out = (-1.0 * h_r_out) * (test_rs[tr_outmask] - r0)

            midprof[i] += np.mean(np.append(x_in, x_out))

        midprof = np.interp(r_data[midmask], midrs, midprof)

    else:
        midprof = np.array([])

    out = np.append(np.append(inprof, midprof), outprof)
    return np.exp(out[reverse_sort])


def radial_broken_exp(
    h_r_in: float,
    h_r_out: float,
    r_peak: float,
    r_data: NDArray,
    norm: float = 1.0,
    r_sun: float = 8.122,
) -> NDArray:
    """Radially Broken Exponential for MW modeling.
    Inputs:
        h_r_in: float: 1/inner exponetial parameter
        h_r_out: float: 1/outer exponetial parameter
        r_peak: float: break radius (kpc)
        r_data: array: array of radii values to evaluate at (kpc)
        norm: optional, float: assumed value at Solar Radius (default 1.0)
    Outputs:
        radial profile for array r_data
    """
    r0 = r_sun
    r_data = np.abs(r_data)
    sort_index = np.argsort(r_data)  # sort for faster comp time
    reverse_sort = np.argsort(sort_index)  # unsort for later
    r_data = r_data[sort_index]

    rp_in = (-1.0 * h_r_in) * (r_peak - r0)
    rp_out = (-1.0 * h_r_out) * (r_peak - r0)
    norm_constant = rp_in - rp_out

    inmask = r_data <= r_peak
    outmask = np.invert(inmask)

    inprof = ((-1.0 * h_r_in) * (r_data[inmask] - r0)) - norm_constant
    outprof = (-1.0 * h_r_out) * (r_data[outmask] - r0)

    out = np.exp(np.append(inprof, outprof)[reverse_sort])

    if r_peak <= r0:
        # x = ((-1.0*h_r_in)*(r_data[inmask]-r0))-norm_constant
        scale = norm / np.exp(-1 * norm_constant)
    else:
        scale = norm

    return scale * out


def z_profile_exp(
    h_z0: float,
    r_flare: float,
    r_data: NDArray,
    z_data: NDArray,
    r_sun: float = 8.122,
) -> NDArray:
    """Single Vertical exponential for MW modeling.
    Inputs:
        h_z0: float: scale height
        R: float: array of radii values to evaluate at (kpc)
        Z: array: array of z values to evaluate at (kpc)
        norm: optional, float: assumed value at Solar Radius (default 1.0)
    Outputs:
        profile for array (r)x(z)
    """
    # r0 = r_sun
    z_data = np.abs(z_data)
    r_data = np.abs(r_data)

    out = np.zeros((len(r_data), len(z_data)))

    for ir, r in enumerate(r_data):
        out[ir] = (-1.0 / h_z0) * (np.abs(z_data))

    return np.exp(out)


def z_profile_flaring(
    h_z0: float,
    r_flare: float,
    r_data: NDArray,
    z_data: NDArray,
    norm: float = 1.0,
    r_sun: float = 8.122,
) -> NDArray:
    """Vertical exponential for MW modeling.
    Inputs:
        h_z0: float: scale height
        r_flare: float: 1/scale radius for flaring
        R: float: array of radii values to evaluate at (kpc)
        Z: array: array of z values to evaluate at (kpc)
        norm: optional, float: assumed value at Solar Radius (default 1.0)
    Outputs:
        profile for array (r)x(z)
    """
    r0 = r_sun
    z_data = np.abs(z_data)
    r_data = np.abs(r_data)
    out = (-1.0 / h_z0) * np.exp((-1.0 * r_flare) * (r_data - r0)) * z_data
    return norm * np.exp(out)


def z_profile_linear(
    h_z0: float,
    r_flare: float,
    r_data: NDArray,
    z_data: NDArray,
    norm: float = 1.0,
    r_sun: float = 8.122,
) -> NDArray:
    """Vertical linear profile for MW modeling.
    Inputs:
        h_z0: float: scale height
        r_flare: float: scale radius for flaring
        R: float: array of radii values to evaluate at (kpc)
        Z: array: array of z values to evaluate at (kpc)
        norm: optional, float: assumed value at Solar Radius (default 1.0)
    Outputs:
        profile for array (r)x(z)
    """
    r0 = r_sun
    z_data = np.abs(z_data)
    r_data = np.abs(r_data)
    h_z = (r_flare * (r_data - r0)) + h_z0
    out = (-1.0 / h_z) * z_data
    return np.exp(out)


def combined_exp_profile_smoothed(
    params: NDArray, r_points: NDArray, z_points: NDArray, r_sun: float = 8.122
) -> NDArray:
    """Combined r,z profile for MW modeling.
    Inputs:
        r_points: array: array of radii values to evaluate at (kpc)
        z_points: array: array of z values to evaluate at (kpc)
        hri: float: inner exponetial parameter
        hro: float: outer exponetial parameter
        hz0: float: scale height (kpc)
        rp: float: radial break radius (kpc)
        rf: float: scale radius for flaring (kpc)
        norm: optional, float: assumed/scale value at
            Solar Radius (default 1.0)
    Outputs:
        array; len(r); density profile for all points (r,z)
    """
    if len(params) == 5:
        hri, hro, hz0, rp, rf = params
        norm = 1.0
    else:
        hri, hro, hz0, rp, rf, norm = params

    # Normalization
    r0 = r_sun
    # z_points=np.abs(z_points)
    # r_points=np.abs(r_points)

    r_norm = radial_broken_exp_smoothed(hri, hro, rp, np.array([r0]))[0]
    # z_norm = z_profile_flaring(hz0,rf,[r0],[0])[0]
    z_norm = 1.0

    scale = norm / (r_norm * z_norm)

    rm = radial_broken_exp_smoothed(hri, hro, rp, r_points)
    zm = z_profile_flaring(hz0, rf, r_points, z_points)
    # zm = (-1.0/hz0)*np.exp((-1.0/rf)*(r_points-r0))*z_points
    # zm = np.exp(zm)

    out = rm * zm * scale

    return out


def combined_exp_profile_nonsmoothed(
    params: NDArray, r_points: NDArray, z_points: NDArray, r_sun: float = 8.122
) -> NDArray:
    """Combined r,z profile for MW modeling.
    Inputs:
        r_points: array: array of radii values to evaluate at (kpc)
        z_points: array: array of z values to evaluate at (kpc)
        hri: float: inner exponetial parameter
        hro: float: outer exponetial parameter
        hz0: float: scale height (kpc)
        rp: float: radial break radius (kpc)
        rf: float: scale radius for flaring (kpc)
        norm: optional, float: assumed/scale value at
            Solar Radius (default 1.0)
    Outputs:
        array; len(r); density profile for all points (r,z)
    """
    if len(params) == 5:
        hri, hro, hz0, rp, rf = params
        norm = 1.0
    else:
        hri, hro, hz0, rp, rf, norm = params
        norm = 10.0**norm

    # Normalization
    r0 = r_sun
    # z_points=np.abs(z_points)
    # r_points=np.abs(r_points)

    r_norm = radial_broken_exp(hri, hro, rp, np.array([r0]))[0]
    # z_norm = z_profile_flaring(hz0,rf,[r0],[0])[0]
    z_norm = 1.0

    scale = norm / (r_norm * z_norm)

    rm = radial_broken_exp(hri, hro, rp, r_points)
    zm = z_profile_flaring(hz0, rf, r_points, z_points)
    # zm = (-1.0/hz0)*np.exp((-1.0/rf)*(r_points-r0))*z_points
    # zm = np.exp(zm)

    out = rm * zm * scale

    return out


def combined_exp_profile_nonsmoothed_linear(
    params: NDArray, r_points: NDArray, z_points: NDArray, r_sun: float = 8.122
) -> NDArray:
    """Combined r,z profile for MW modeling.
    Inputs:
        r_points: array: array of radii values to evaluate at (kpc)
        z_points: array: array of z values to evaluate at (kpc)
        hri: float: inner exponetial parameter
        hro: float: outer exponetial parameter
        hz0: float: scale height (kpc)
        rp: float: radial break radius (kpc)
        rf: float: scale radius for flaring (kpc)
        norm: optional, float: assumed/scale value at
            Solar Radius (default 1.0)
    Outputs:
        array; len(r); density profile for all points (r,z)
    """
    if len(params) == 5:
        hri, hro, hz0, rp, rf = params
        norm = 1.0
    else:
        hri, hro, hz0, rp, rf, norm = params
        norm = 10.0**norm

    # Normalization
    r0 = r_sun
    # z_points=np.abs(z_points)
    # r_points=np.abs(r_points)

    r_norm = radial_broken_exp(hri, hro, rp, np.array([r0]))[0]
    z_norm = z_profile_linear(hz0, rf, np.array([r0]), np.array([0]))[0]
    # z_norm = 1.0

    scale = norm / (r_norm * z_norm)

    rm = radial_broken_exp(hri, hro, rp, r_points)
    zm = z_profile_linear(hz0, rf, r_points, z_points)
    # zm = (-1.0/hz0)*np.exp((-1.0/rf)*(r_points-r0))*z_points
    # zm = np.exp(zm)

    out = rm * zm * scale

    return out


def uniform_profile(
    params: NDArray, r_points: NDArray, z_points: NDArray, r_sun: float = 8.122
) -> NDArray:
    """
    A flat/uniform density profile, independent of R and Z values.
    Used only for testing.
    Inputs:
        params:        norm: optional, float: assumed/scale value
             at Solar Radius (default 1.0)
    Outputs:
        array; len(r); density profile for all points (r,z)
    """

    norm = np.median(params)
    norm = 10.0**norm
    out = [norm for p in r_points]
    return np.array(out)


def z_indep_profile(
    params: NDArray, r_points: NDArray, z_points: NDArray, r_sun: float = 8.122
) -> NDArray:
    """
    Z independent version of combined_exp_profile_nonsmoothed_linear
    for testing purposes
    """
    if len(params) == 5:
        hri, hro, hz0, rp, rf = params
        norm = 1.0
    else:
        hri, hro, hz0, rp, rf, norm = params
        norm = 10.0**norm

    # Normalization
    r0 = r_sun
    # z_points=np.abs(z_points)
    # r_points=np.abs(r_points)

    r_norm = radial_broken_exp(hri, hro, rp, np.array([r0]))[0]
    # z_norm = z_profile_linear(hz0,rf,[r0],[0])[0]
    # z_norm = 1.0

    scale = norm / (r_norm)

    rm = radial_broken_exp(hri, hro, rp, r_points)
    # zm = z_profile_linear(hz0,rf,r_points,z_points)
    # zm = (-1.0/hz0)*np.exp((-1.0/rf)*(r_points-r0))*z_points
    # zm = np.exp(zm)

    out = rm * scale

    return out
