"""
Neoclassical conductivity and bootstrap current coefficients.

Implements the analytic fits of

    O. Sauter, C. Angioni and Y. R. Lin-Liu,
    "Neoclassical conductivity and bootstrap current formulas for general
    axisymmetric equilibria and arbitrary collisionality regime",
    Phys. Plasmas 6, 2834 (1999)

Equation numbers in the comments below refer to that paper. The functions
here are deliberately pure: they take trapped fraction, collisionalities and
effective charge and return dimensionless coefficients, with no dependence on
the equilibrium representation. That makes them directly comparable against
the figures and asymptotic limits quoted in the paper.

The flux surface averaged parallel current is (Eq. 5, and the equivalent form
in the Conclusion)

    <j.B> = sigma_neo * <E.B>
            - I(psi) * p_e * [ L31 * (p/p_e) * dln(p)/dpsi
                               + L32 * dln(Te)/dpsi
                               + L34 * alpha * dln(Ti)/dpsi ]

where I(psi) = R * Btor = f.

Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGS.

FreeGS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGS.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

#: Elementary charge [C]. Temperatures in this module are in eV, so this is
#: also the conversion factor from eV to Joules.
ELEMENTARY_CHARGE = 1.602176634e-19

#: Smallest Coulomb logarithm considered physical. The fits in Eq. (18d,e)
#: are unbounded below and can go negative for absurd n/T combinations; the
#: floor keeps sigma_Sptz positive and nu* finite.
MIN_COULOMB_LOG = 1.0

#: Trapped fraction is clipped into this range. f_t = 1 would make the
#: collisionality-corrected denominators meaningless and f_t < 0 is unphysical.
_FT_MIN = 0.0
_FT_MAX = 0.9999


def _as_array(x):
    """Return x as a float array, preserving scalars as 0-d arrays."""
    return np.asarray(x, dtype=float)


def coulomb_log_e(ne, Te):
    """
    Electron Coulomb logarithm, Eq. (18d).

    ln Lambda_e = 31.3 - ln( sqrt(ne) / Te )

    Parameters
    ----------
    ne:
        Electron density [m^-3]
    Te:
        Electron temperature [eV]

    Returns
    -------
    Coulomb logarithm, floored at MIN_COULOMB_LOG.
    """
    ne = _as_array(ne)
    Te = _as_array(Te)
    return np.maximum(31.3 - np.log(np.sqrt(ne) / Te), MIN_COULOMB_LOG)


def coulomb_log_ii(ni, Ti, Z):
    """
    Ion-ion Coulomb logarithm, Eq. (18e).

    ln Lambda_ii = 30 - ln( Z^3 * sqrt(ni) / Ti^(3/2) )

    Parameters
    ----------
    ni:
        Ion density [m^-3]
    Ti:
        Ion temperature [eV]
    Z:
        Ion charge

    Returns
    -------
    Coulomb logarithm, floored at MIN_COULOMB_LOG.
    """
    ni = _as_array(ni)
    Ti = _as_array(Ti)
    Z = _as_array(Z)
    return np.maximum(30.0 - np.log(Z**3 * np.sqrt(ni) / Ti**1.5), MIN_COULOMB_LOG)


def spitzer_conductivity(Te, Z, coulomb_log=None, ne=None):
    """
    Spitzer conductivity, Eq. (18a).

    sigma_Sptz = 1.9012e4 * Te[eV]^(3/2) / ( Z * N(Z) * ln Lambda_e )

    with N(Z) = 0.58 + 0.74 / (0.76 + Z).

    Parameters
    ----------
    Te:
        Electron temperature [eV]
    Z:
        Effective charge
    coulomb_log:
        ln Lambda_e. If None, computed from ne and Te via Eq. (18d),
        in which case ne must be given.
    ne:
        Electron density [m^-3], only used if coulomb_log is None.

    Returns
    -------
    Conductivity [Ohm^-1 m^-1]
    """
    Te = _as_array(Te)
    Z = _as_array(Z)

    if coulomb_log is None:
        if ne is None:
            raise ValueError("spitzer_conductivity needs either coulomb_log or ne")
        coulomb_log = coulomb_log_e(ne, Te)

    NZ = 0.58 + 0.74 / (0.76 + Z)
    return 1.9012e4 * Te**1.5 / (Z * NZ * coulomb_log)


def nu_star_e(ne, Te, Z, q, R, eps, coulomb_log=None):
    """
    Electron collisionality, Eq. (18b).

    nu*_e = 6.921e-18 * q * R * ne * Z * ln Lambda_e / ( Te^2 * eps^(3/2) )

    Parameters
    ----------
    ne:
        Electron density [m^-3]
    Te:
        Electron temperature [eV]
    Z:
        Effective charge
    q:
        Safety factor. The absolute value is used, so either sign
        convention for the plasma current is acceptable.
    R:
        Major radius of the flux surface [m]
    eps:
        Inverse aspect ratio of the flux surface
    coulomb_log:
        ln Lambda_e. Computed from Eq. (18d) if not given.
    """
    ne = _as_array(ne)
    Te = _as_array(Te)

    if coulomb_log is None:
        coulomb_log = coulomb_log_e(ne, Te)

    return (
        6.921e-18 * np.abs(q) * R * ne * Z * coulomb_log / (Te**2 * np.power(eps, 1.5))
    )


def nu_star_i(ni, Ti, Z, q, R, eps, coulomb_log=None):
    """
    Ion collisionality, Eq. (18c).

    nu*_i = 4.90e-18 * q * R * ni * Z^4 * ln Lambda_ii / ( Ti^2 * eps^(3/2) )

    Parameters
    ----------
    ni:
        Ion density [m^-3]
    Ti:
        Ion temperature [eV]
    Z:
        Ion charge
    q:
        Safety factor (absolute value used)
    R:
        Major radius of the flux surface [m]
    eps:
        Inverse aspect ratio of the flux surface
    coulomb_log:
        ln Lambda_ii. Computed from Eq. (18e) if not given.
    """
    ni = _as_array(ni)
    Ti = _as_array(Ti)
    Z = _as_array(Z)

    if coulomb_log is None:
        coulomb_log = coulomb_log_ii(ni, Ti, Z)

    return (
        4.90e-18
        * np.abs(q)
        * R
        * ni
        * Z**4
        * coulomb_log
        / (Ti**2 * np.power(eps, 1.5))
    )


# ----------------------------------------------------------------------------
# Effective trapped fractions, Eqs. (13b), (14b), (15d), (15e), (16b)
#
# Sauter's key observation (Sec. III) is that finite collisionality can be
# folded entirely into an effective trapped fraction, leaving the polynomial
# coefficients collisionality-independent.
# ----------------------------------------------------------------------------


def ft_eff_33(ft, nue_star, Z):
    """Effective trapped fraction for the conductivity, Eq. (13b)."""
    return ft / (
        1.0
        + (0.55 - 0.1 * ft) * np.sqrt(nue_star)
        + 0.45 * (1.0 - ft) * nue_star / Z**1.5
    )


def ft_eff_31(ft, nue_star, Z):
    """Effective trapped fraction for L31, Eq. (14b)."""
    return ft / (
        1.0 + (1.0 - 0.1 * ft) * np.sqrt(nue_star) + 0.5 * (1.0 - ft) * nue_star / Z
    )


def ft_eff_32_ee(ft, nue_star, Z):
    """Effective trapped fraction for the L32 electron-electron term, Eq. (15d)."""
    return ft / (
        1.0
        + 0.26 * (1.0 - ft) * np.sqrt(nue_star)
        + 0.18 * (1.0 - 0.37 * ft) * nue_star / np.sqrt(Z)
    )


def ft_eff_32_ei(ft, nue_star, Z):
    """Effective trapped fraction for the L32 electron-ion term, Eq. (15e)."""
    return ft / (
        1.0
        + (1.0 + 0.6 * ft) * np.sqrt(nue_star)
        + 0.85 * (1.0 - 0.37 * ft) * nue_star * (1.0 + Z)
    )


def ft_eff_34(ft, nue_star, Z):
    """Effective trapped fraction for L34, Eq. (16b).

    Note this differs from Eq. (14b) only in the last term
    (0.5*(1 - 0.5*ft) rather than 0.5*(1 - ft)), which is why L34 ~ L31
    except at very large collisionality (Fig. 6).
    """
    return ft / (
        1.0
        + (1.0 - 0.1 * ft) * np.sqrt(nue_star)
        + 0.5 * (1.0 - 0.5 * ft) * nue_star / Z
    )


# ----------------------------------------------------------------------------
# Polynomial fits, Eqs. (13a), (14a), (15b), (15c)
# ----------------------------------------------------------------------------


def F33(X, Z):
    """
    Conductivity polynomial, Eq. (13a).

    sigma_neo / sigma_Sptz = 1 - (1 + 0.36/Z) X + (0.59/Z) X^2 - (0.23/Z) X^3
    """
    return 1.0 - (1.0 + 0.36 / Z) * X + (0.59 / Z) * X**2 - (0.23 / Z) * X**3


def F31(X, Z):
    """
    L31 (and L34) polynomial, Eq. (14a).

    F31 = (1 + 1.4/(Z+1)) X - (1.9/(Z+1)) X^2 + (0.3/(Z+1)) X^3 + (0.2/(Z+1)) X^4
    """
    Zp1 = Z + 1.0
    return (
        (1.0 + 1.4 / Zp1) * X
        - (1.9 / Zp1) * X**2
        + (0.3 / Zp1) * X**3
        + (0.2 / Zp1) * X**4
    )


def F32_ee(X, Z):
    """L32 electron-electron contribution, Eq. (15b)."""
    return (
        (0.05 + 0.62 * Z) / (Z * (1.0 + 0.44 * Z)) * (X - X**4)
        + 1.0 / (1.0 + 0.22 * Z) * (X**2 - X**4 - 1.2 * (X**3 - X**4))
        + 1.2 / (1.0 + 0.5 * Z) * X**4
    )


def F32_ei(Y, Z):
    """L32 electron-ion contribution, Eq. (15c)."""
    return (
        -(0.56 + 1.93 * Z) / (Z * (1.0 + 0.44 * Z)) * (Y - Y**4)
        + 4.95 / (1.0 + 2.48 * Z) * (Y**2 - Y**4 - 0.55 * (Y**3 - Y**4))
        - 1.2 / (1.0 + 0.5 * Z) * Y**4
    )


# ----------------------------------------------------------------------------
# Public coefficient functions
# ----------------------------------------------------------------------------


def sigma_neo(ft, nue_star, Z, Te, ne=None, coulomb_log=None):
    """
    Neoclassical parallel conductivity, Eqs. (13) and (18a).

    Parameters
    ----------
    ft:
        Trapped particle fraction
    nue_star:
        Electron collisionality, Eq. (18b)
    Z:
        Effective charge
    Te:
        Electron temperature [eV]
    ne:
        Electron density [m^-3], used for ln Lambda_e if coulomb_log is None
    coulomb_log:
        ln Lambda_e

    Returns
    -------
    sigma_neo [Ohm^-1 m^-1]
    """
    ft = np.clip(_as_array(ft), _FT_MIN, _FT_MAX)
    X = ft_eff_33(ft, nue_star, Z)
    return F33(X, Z) * spitzer_conductivity(Te, Z, coulomb_log=coulomb_log, ne=ne)


# ----------------------------------------------------------------------------
# Sign convention
# ----------------------------------------------------------------------------
#
# The polynomials above are transcribed exactly as printed in Eqs. (14a),
# (15b) and (15c), so they can be checked against the paper directly. As
# printed they give F31 > 0 and F32_ee + F32_ei < 0 for typical arguments, for
# example F31 = +0.60 and F32_ee + F32_ei = -0.17 at f_t = 0.5, Z = 1.8.
#
# That is the opposite sign to the values the paper itself quotes in its
# Conclusion, "L31 ~ L34 ~ -0.5, L32 ~ 0.2, alpha ~ -0.5", which are the
# values that reproduce the drive coefficients also quoted there (-0.5 for the
# density gradient, -0.15 for T_e, -0.1 for T_i at R_pe = 0.5). The
# discrepancy is a sign convention for psi: Eq. (5) is written for the
# opposite orientation of the flux coordinate to the one implied by the fits.
#
# Resolving it by inspection of the paper alone is ambiguous, so the sign is
# fixed here by physics instead: the bootstrap current must be co-current,
# that is it must add to Ip, when the pressure falls outwards. Combining
# Eq. (5) with the FreeGS convention psi_bndry - psi_axis = -sign(Ip)|dpsi|
# and Jtor_bs = f <j.B>_bs / (R <B^2>), the toroidal bootstrap current works
# out proportional to -sign(psi_bndry - psi_axis) = +sign(Ip), independent of
# the sign of f, only if L31, L32 and L34 are the negatives of the printed
# polynomials. Checking all four combinations of the signs of Ip and f
# confirms that choice is the only self-consistent one.
#
# The coefficients returned below therefore match the paper's quoted values,
# and are what should be compared against its Figs. 3 and 6.


def L31(ft, nue_star, Z):
    """Bootstrap coefficient L31 (pressure gradient drive), Eq. (14).

    Negative for typical parameters, matching the paper's quoted L31 ~ -0.5.
    """
    ft = np.clip(_as_array(ft), _FT_MIN, _FT_MAX)
    return -F31(ft_eff_31(ft, nue_star, Z), Z)


def L32(ft, nue_star, Z):
    """Bootstrap coefficient L32 (electron temperature gradient drive), Eq. (15).

    Split internally into the ee and ei contributions of Eq. (9), which have
    different collisionality dependences (Figs. 4 and 5) and opposite signs.

    Positive for typical parameters, matching the paper's quoted L32 ~ 0.2.
    """
    ft = np.clip(_as_array(ft), _FT_MIN, _FT_MAX)
    X = ft_eff_32_ee(ft, nue_star, Z)
    Y = ft_eff_32_ei(ft, nue_star, Z)
    return -(F32_ee(X, Z) + F32_ei(Y, Z))


def L34(ft, nue_star, Z):
    """Bootstrap coefficient L34 (ion temperature gradient drive), Eq. (16).

    Negative for typical parameters, matching the paper's quoted L34 ~ -0.5.
    """
    ft = np.clip(_as_array(ft), _FT_MIN, _FT_MAX)
    return -F31(ft_eff_34(ft, nue_star, Z), Z)


def alpha(ft, nui_star):
    """
    Ion temperature gradient coefficient alpha, Eq. (17).

    alpha_0 = -1.17 (1 - ft) / (1 - 0.22 ft - 0.19 ft^2)

    alpha = [ (alpha_0 + 0.25 (1 - ft^2) sqrt(nui*)) / (1 + 0.5 sqrt(nui*))
              - 0.315 nui*^2 ft^6 ] / (1 + 0.15 nui*^2 ft^6)

    Unlike the other coefficients this is not written as a polynomial in an
    effective trapped fraction: the paper notes (Sec. III, Fig. 7) that the
    ft and nui* dependences cannot be decoupled here.

    Note that the actual coefficient multiplying dln(Ti)/dpsi is L34 * alpha.
    """
    ft = np.clip(_as_array(ft), _FT_MIN, _FT_MAX)
    nui_star = _as_array(nui_star)

    alpha0 = -1.17 * (1.0 - ft) / (1.0 - 0.22 * ft - 0.19 * ft**2)

    sq = np.sqrt(nui_star)
    ft6 = ft**6
    nui2ft6 = nui_star**2 * ft6

    return (
        (alpha0 + 0.25 * (1.0 - ft**2) * sq) / (1.0 + 0.5 * sq) - 0.315 * nui2ft6
    ) / (1.0 + 0.15 * nui2ft6)


def bootstrap_coefficients(ft, nue_star, nui_star, Zeff):
    """
    Evaluate all four bootstrap coefficients at once.

    Parameters
    ----------
    ft:
        Trapped particle fraction
    nue_star, nui_star:
        Electron and ion collisionalities, Eqs. (18b) and (18c)
    Zeff:
        Effective charge

    Returns
    -------
    dict with keys "L31", "L32", "L34", "alpha"
    """
    return {
        "L31": L31(ft, nue_star, Zeff),
        "L32": L32(ft, nue_star, Zeff),
        "L34": L34(ft, nue_star, Zeff),
        "alpha": alpha(ft, nui_star),
    }


def jdotB_bootstrap(
    f,
    pe,
    p,
    dlnp_dpsi,
    dlnTe_dpsi,
    dlnTi_dpsi,
    ft,
    nue_star,
    nui_star,
    Zeff,
):
    """
    Flux surface averaged bootstrap contribution to <j.B>, Eq. (5).

    <j.B>_BS = - I(psi) * p_e * [ L31 * A1 + L32 * A2 + L34 * A4 ]

    with the drives of Eq. (6),

        A1 = (1/p_e) dp/dpsi              = (p/p_e)  dln(p)/dpsi
        A2 =                                          dln(Te)/dpsi
        A4 = alpha * ((1 - R_pe)/R_pe) * dln(Ti)/dpsi,   R_pe = p_e/p

    Note the (1 - R_pe)/R_pe = p_i/p_e factor on the ion temperature drive.
    The abbreviated form written in the paper's Conclusion omits it; it is
    kept here because Eq. (6) defines it and because it is needed for the two
    forms given in the Conclusion to agree with each other. It only makes no
    difference in the special case R_pe = 1/2.

    Substituting R_pe and multiplying out,

    <j.B>_BS = - I(psi) * [ L31 * p * dln(p)/dpsi
                            + L32 * p_e * dln(Te)/dpsi
                            + L34 * alpha * (p - p_e) * dln(Ti)/dpsi ]

    All gradients are with respect to the *signed* poloidal flux psi, so this
    expression is correct for either sign of plasma current and toroidal
    field; see the sign convention note above.

    Parameters
    ----------
    f:
        I(psi) = R * Btor [T m]
    pe:
        Electron pressure [Pa]
    p:
        Total pressure [Pa]
    dlnp_dpsi, dlnTe_dpsi, dlnTi_dpsi:
        Logarithmic gradients with respect to psi [Wb^-1]
    ft, nue_star, nui_star, Zeff:
        As for bootstrap_coefficients

    Returns
    -------
    <j.B>_BS [A T m^-2]
    """
    c = bootstrap_coefficients(ft, nue_star, nui_star, Zeff)

    return -f * (
        c["L31"] * p * dlnp_dpsi
        + c["L32"] * pe * dlnTe_dpsi
        + c["L34"] * c["alpha"] * (p - pe) * dlnTi_dpsi
    )


def jdotB_ohmic(sigma, f, R2inv_avg, Vloop):
    """
    Flux surface averaged ohmic contribution to <j.B>.

    <j.B>_ohm = sigma_neo * <E.B>

    with the inductive parallel electric field written in terms of the loop
    voltage. Since E_phi = Vloop / (2 pi R) and B_phi = f / R,

        <E.B> = Vloop * f * <1/R^2> / (2 pi)

    Parameters
    ----------
    sigma:
        Neoclassical conductivity [Ohm^-1 m^-1]
    f:
        I(psi) = R * Btor [T m]
    R2inv_avg:
        Flux surface average <1/R^2> [m^-2]
    Vloop:
        Loop voltage [V]

    Returns
    -------
    <j.B>_ohm [A T m^-2]
    """
    return sigma * Vloop * f * R2inv_avg / (2.0 * np.pi)
