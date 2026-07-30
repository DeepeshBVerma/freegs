"""
Tests for the Sauter neoclassical coefficient fits.

These check the formulas against the limits and values quoted in

    O. Sauter, C. Angioni and Y. R. Lin-Liu,
    Phys. Plasmas 6, 2834 (1999)
"""

import numpy as np
import pytest

from freegs import sauter

# ----------------------------------------------------------------------------
# Conductivity
# ----------------------------------------------------------------------------


def test_F33_collisionless_no_trapping():
    """With no trapped particles the neoclassical conductivity is Spitzer."""
    for Z in (1.0, 2.0, 5.0):
        assert sauter.F33(0.0, Z) == pytest.approx(1.0)


def test_F33_decreases_with_trapping():
    """Trapped particles reduce the conductivity below Spitzer."""
    ft = np.linspace(0.0, 0.9, 20)
    for Z in (1.0, 2.0, 4.0):
        ratio = sauter.F33(ft, Z)
        assert np.all(np.diff(ratio) < 0.0)
        assert np.all(ratio <= 1.0)
        assert np.all(ratio > 0.0)


def test_sigma_neo_reduces_to_spitzer_without_trapping():
    Te, ne, Z = 1000.0, 5.0e19, 1.5
    sig = sauter.sigma_neo(0.0, 0.1, Z, Te, ne=ne)
    assert sig == pytest.approx(sauter.spitzer_conductivity(Te, Z, ne=ne))


def test_spitzer_temperature_scaling():
    """sigma_Sptz scales as Te^(3/2) at fixed Coulomb logarithm."""
    lnL = 17.0
    s1 = sauter.spitzer_conductivity(1000.0, 1.0, coulomb_log=lnL)
    s2 = sauter.spitzer_conductivity(4000.0, 1.0, coulomb_log=lnL)
    assert s2 / s1 == pytest.approx(4.0**1.5)


def test_spitzer_NZ_factor():
    """N(Z) = 0.58 + 0.74/(0.76 + Z), Eq. (18a)."""
    Te, lnL = 1000.0, 17.0
    for Z in (1.0, 2.0, 3.0):
        NZ = 0.58 + 0.74 / (0.76 + Z)
        expected = 1.9012e4 * Te**1.5 / (Z * NZ * lnL)
        assert sauter.spitzer_conductivity(Te, Z, coulomb_log=lnL) == pytest.approx(
            expected
        )


def test_coulomb_logs_are_positive_and_floored():
    """Absurd n/T must not produce a negative Coulomb logarithm."""
    assert sauter.coulomb_log_e(1.0e22, 0.1) >= sauter.MIN_COULOMB_LOG
    assert sauter.coulomb_log_ii(1.0e22, 0.1, 10.0) >= sauter.MIN_COULOMB_LOG
    # Reasonable tokamak values should be in the usual range
    assert 10.0 < sauter.coulomb_log_e(5.0e19, 2000.0) < 20.0
    assert 10.0 < sauter.coulomb_log_ii(5.0e19, 2000.0, 1.0) < 25.0


# ----------------------------------------------------------------------------
# Bootstrap coefficients: values quoted in the paper's Conclusion
# ----------------------------------------------------------------------------


def test_conclusion_coefficient_values():
    """
    The Conclusion states L31 ~ L34 ~ -0.5, L32 ~ 0.2 and alpha ~ -0.5.

    This pins the sign convention: see the note in sauter.py. The values are
    reproduced near ft = 0.5 in the banana regime.
    """
    ft, nue, nui, Z = 0.5, 0.01, 0.01, 1.0

    l31 = sauter.L31(ft, nue, Z)
    l32 = sauter.L32(ft, nue, Z)
    l34 = sauter.L34(ft, nue, Z)
    a = sauter.alpha(ft, nui)

    assert l31 == pytest.approx(-0.5, abs=0.12)
    assert l34 == pytest.approx(-0.5, abs=0.12)
    assert l32 == pytest.approx(0.2, abs=0.05)
    assert a == pytest.approx(-0.5, abs=0.16)

    # Signs must be definite, not just close in magnitude
    assert l31 < 0.0
    assert l34 < 0.0
    assert l32 > 0.0
    assert a < 0.0


def test_conclusion_drive_coefficients():
    """
    The Conclusion also quotes the drive coefficients in the form

        <j.B> = sigma <E.B> - I p [ c_n dln(ne)/dpsi + c_Te dln(Te)/dpsi
                                    + c_Ti dln(Ti)/dpsi ]

    with c_n ~ -0.5, c_Te ~ -0.15, c_Ti ~ -0.1 at R_pe = 0.5, where
    c_n = L31, c_Te = R_pe (L31 + L32) and
    c_Ti = (1 - R_pe)(1 + (L34/L31) alpha) L31.
    """
    ft, nue, nui, Z, Rpe = 0.5, 0.01, 0.01, 1.0, 0.5

    l31 = sauter.L31(ft, nue, Z)
    l32 = sauter.L32(ft, nue, Z)
    l34 = sauter.L34(ft, nue, Z)
    a = sauter.alpha(ft, nui)

    c_n = l31
    c_Te = Rpe * (l31 + l32)
    c_Ti = (1.0 - Rpe) * (1.0 + (l34 / l31) * a) * l31

    assert c_n == pytest.approx(-0.5, abs=0.12)
    assert c_Te == pytest.approx(-0.15, abs=0.06)
    assert c_Ti == pytest.approx(-0.1, abs=0.05)


def test_L34_close_to_L31_in_banana_regime():
    """
    The paper notes L34 = L31 in the collisionless limit, and that they only
    differ at very large collisionality (Fig. 6).
    """
    ft = np.linspace(0.05, 0.9, 20)
    for Z in (1.0, 2.0, 3.0):
        assert np.allclose(
            sauter.L31(ft, 1.0e-6, Z), sauter.L34(ft, 1.0e-6, Z), rtol=1e-6
        )
        # At high collisionality they must diverge
        assert not np.allclose(
            sauter.L31(ft, 50.0, Z), sauter.L34(ft, 50.0, Z), rtol=1e-3
        )


def test_coefficients_vanish_without_trapped_particles():
    """No trapped particles means no bootstrap current."""
    for Z in (1.0, 2.0, 4.0):
        assert sauter.L31(0.0, 0.1, Z) == pytest.approx(0.0)
        assert sauter.L32(0.0, 0.1, Z) == pytest.approx(0.0)
        assert sauter.L34(0.0, 0.1, Z) == pytest.approx(0.0)


def test_L31_magnitude_grows_with_trapped_fraction():
    ft = np.linspace(0.02, 0.9, 30)
    for Z in (1.0, 2.0, 3.0):
        l31 = sauter.L31(ft, 0.01, Z)
        assert np.all(np.diff(l31) < 0.0)  # becoming more negative


def test_collisionality_suppresses_bootstrap():
    """Increasing nu*_e must reduce the bootstrap drive towards zero."""
    ft, Z = 0.6, 1.5
    nue = np.array([1.0e-4, 1.0e-2, 1.0, 10.0, 100.0])
    l31 = sauter.L31(ft, nue, Z)
    assert np.all(np.diff(np.abs(l31)) < 0.0)
    assert abs(l31[-1]) < 0.1 * abs(l31[0])


def test_L32_changes_sign_with_collisionality():
    """
    Section III: L32 = L32_ee + L32_ei is a sum of terms of opposite sign with
    different collisionality dependence, and "even changes sign" (Fig. 5).
    """
    ft, Z = 0.5, 1.0
    nue = np.logspace(-4, 2, 200)
    l32 = sauter.L32(ft, nue, Z)
    assert np.any(l32 > 0.0)
    assert np.any(l32 < 0.0)


def test_L32_vanishes_at_high_collisionality():
    """Section III: "the limit at high collisionality of L32 is zero"."""
    assert abs(sauter.L32(0.5, 1.0e4, 1.0)) < 0.02


def test_alpha_banana_limit():
    """alpha_0 = -1.17 (1 - ft) / (1 - 0.22 ft - 0.19 ft^2), Eq. (17a)."""
    ft = np.linspace(0.0, 0.9, 25)
    expected = -1.17 * (1.0 - ft) / (1.0 - 0.22 * ft - 0.19 * ft**2)
    assert np.allclose(sauter.alpha(ft, 0.0), expected)


def test_alpha_is_finite_and_negative_in_banana_regime():
    ft = np.linspace(0.0, 0.95, 30)
    nui = np.logspace(-4, 2, 30)
    F, N = np.meshgrid(ft, nui)
    a = sauter.alpha(F, N)
    assert np.all(np.isfinite(a))

    # In the banana regime alpha is negative for every trapped fraction
    assert np.all(sauter.alpha(ft, 1.0e-4) < 0.0)

    # Fig. 7(a): increasing nu*_i makes alpha sharply more negative once
    # there is appreciable trapping
    assert sauter.alpha(0.6, 10.0) < sauter.alpha(0.6, 0.01)


def test_alpha_sign_change_at_low_trapping_is_harmless():
    """
    Eq. (17b) contains a +0.25 (1 - ft^2) sqrt(nui*) term which is not damped
    by the ft^6 factors, so alpha turns slightly positive for small ft at very
    high nui*. That is a property of the fit, not an error, and it cannot
    affect the current because the coefficient multiplying alpha is L34, which
    vanishes as ft -> 0.
    """
    assert sauter.alpha(0.05, 100.0) > 0.0

    # The ion temperature drive coefficient is L34 * alpha, which still goes
    # to zero with the trapped fraction
    for nui in (0.01, 1.0, 100.0):
        drive = sauter.L34(0.0, nui, 1.5) * sauter.alpha(0.0, nui)
        assert drive == pytest.approx(0.0)

    ft_small = 0.02
    drive_small = abs(sauter.L34(ft_small, 100.0, 1.5) * sauter.alpha(ft_small, 100.0))
    assert drive_small < 0.01


def test_effective_trapped_fractions_bounded_by_ft():
    """
    Finite collisionality can only reduce the effective trapped fraction,
    which is the whole point of Sauter's parameterisation (Sec. III).
    """
    ft = np.linspace(0.01, 0.95, 25)
    for Z in (1.0, 2.0, 4.0):
        for nue in (0.0, 0.01, 1.0, 100.0):
            for fn in (
                sauter.ft_eff_33,
                sauter.ft_eff_31,
                sauter.ft_eff_32_ee,
                sauter.ft_eff_32_ei,
                sauter.ft_eff_34,
            ):
                x = fn(ft, nue, Z)
                assert np.all(x <= ft + 1e-12)
                assert np.all(x >= 0.0)
        # Zero collisionality leaves it unchanged
        assert np.allclose(sauter.ft_eff_31(ft, 0.0, Z), ft)


# ----------------------------------------------------------------------------
# Collisionalities
# ----------------------------------------------------------------------------


def test_nu_star_scalings():
    """nu*_e ~ q R n Z lnL / (Te^2 eps^(3/2)), Eq. (18b)."""
    base = {
        "ne": 5.0e19,
        "Te": 2000.0,
        "Z": 1.0,
        "q": 2.0,
        "R": 1.7,
        "eps": 0.3,
        "coulomb_log": 17.0,
    }
    n0 = sauter.nu_star_e(**base)

    assert sauter.nu_star_e(**{**base, "ne": 1.0e20}) / n0 == pytest.approx(2.0)
    assert sauter.nu_star_e(**{**base, "Te": 4000.0}) / n0 == pytest.approx(0.25)
    assert sauter.nu_star_e(**{**base, "q": 4.0}) / n0 == pytest.approx(2.0)
    assert sauter.nu_star_e(**{**base, "eps": 0.3 * 4.0}) / n0 == pytest.approx(
        4.0**-1.5
    )


def test_nu_star_sign_independent_of_q_direction():
    """Negative plasma current gives negative q; nu* must not follow."""
    kw = {
        "ne": 5.0e19,
        "Te": 2000.0,
        "Z": 1.0,
        "R": 1.7,
        "eps": 0.3,
        "coulomb_log": 17.0,
    }
    assert sauter.nu_star_e(q=3.0, **kw) == sauter.nu_star_e(q=-3.0, **kw)

    kwi = {
        "ni": 5.0e19,
        "Ti": 2000.0,
        "Z": 1.0,
        "R": 1.7,
        "eps": 0.3,
        "coulomb_log": 17.0,
    }
    assert sauter.nu_star_i(q=3.0, **kwi) == sauter.nu_star_i(q=-3.0, **kwi)


def test_nu_star_i_charge_scaling():
    """nu*_i ~ Z^4, Eq. (18c)."""
    kw = {
        "ni": 5.0e19,
        "Ti": 2000.0,
        "q": 2.0,
        "R": 1.7,
        "eps": 0.3,
        "coulomb_log": 17.0,
    }
    r = sauter.nu_star_i(Z=2.0, **kw) / sauter.nu_star_i(Z=1.0, **kw)
    assert r == pytest.approx(16.0)


# ----------------------------------------------------------------------------
# <j.B> assembly
# ----------------------------------------------------------------------------


def test_jdotB_bootstrap_sign_is_co_current():
    """
    A pressure profile falling outwards must drive bootstrap current parallel
    to Ip, for every combination of the signs of Ip and f.

    In FreeGS, psi_bndry - psi_axis has the opposite sign to Ip, and the
    toroidal current density of the bootstrap component is
    Jtor_bs = f <j.B>_bs / (R <B^2>). So sign(Jtor_bs) must equal sign(Ip).
    """
    ft, nue, nui, Z = 0.6, 0.05, 0.05, 1.5
    p, pe = 2.0e4, 1.0e4

    # Profiles decreasing outwards: derivatives w.r.t. psi_n are negative
    dlnp_dpsin, dlnTe_dpsin, dlnTi_dpsin = -3.0, -2.0, -2.0

    for Ip_sign in (+1.0, -1.0):
        # FreeGS convention: psi peaks on axis for positive Ip
        dpsi = -Ip_sign * 0.5
        for f in (+3.0, -3.0):
            jdotB = sauter.jdotB_bootstrap(
                f,
                pe,
                p,
                dlnp_dpsin / dpsi,
                dlnTe_dpsin / dpsi,
                dlnTi_dpsin / dpsi,
                ft,
                nue,
                nui,
                Z,
            )
            Jtor_bs = f * jdotB  # positive R and <B^2> do not change the sign
            assert np.sign(Jtor_bs) == Ip_sign, (
                f"Bootstrap current is counter-current for Ip_sign={Ip_sign}, f={f}"
            )


def test_jdotB_bootstrap_vanishes_without_gradients():
    assert sauter.jdotB_bootstrap(
        3.0, 1.0e4, 2.0e4, 0.0, 0.0, 0.0, 0.6, 0.05, 0.05, 1.5
    ) == pytest.approx(0.0)


def test_jdotB_bootstrap_vanishes_without_trapping():
    assert sauter.jdotB_bootstrap(
        3.0, 1.0e4, 2.0e4, -1.0, -1.0, -1.0, 0.0, 0.05, 0.05, 1.5
    ) == pytest.approx(0.0)


def test_jdotB_bootstrap_scales_with_pressure_gradient():
    """The dominant L31 term is linear in the pressure gradient."""
    kw = {"ft": 0.6, "nue_star": 0.05, "nui_star": 0.05, "Zeff": 1.5}
    a = sauter.jdotB_bootstrap(3.0, 1.0e4, 2.0e4, -1.0, 0.0, 0.0, **kw)
    b = sauter.jdotB_bootstrap(3.0, 1.0e4, 2.0e4, -2.0, 0.0, 0.0, **kw)
    assert b / a == pytest.approx(2.0)


def test_jdotB_ohmic_form():
    """<E.B> = Vloop f <1/R^2> / (2 pi)."""
    sigma, f, R2inv, V = 1.0e7, 3.0, 0.4, 0.5
    assert sauter.jdotB_ohmic(sigma, f, R2inv, V) == pytest.approx(
        sigma * V * f * R2inv / (2.0 * np.pi)
    )
    # Linear in Vloop, which the solver relies on to set Ip in closed form
    assert sauter.jdotB_ohmic(sigma, f, R2inv, 2.0 * V) == pytest.approx(
        2.0 * sauter.jdotB_ohmic(sigma, f, R2inv, V)
    )


def test_bootstrap_coefficients_dict():
    c = sauter.bootstrap_coefficients(0.5, 0.01, 0.01, 1.0)
    assert set(c) == {"L31", "L32", "L34", "alpha"}
    assert c["L31"] == pytest.approx(sauter.L31(0.5, 0.01, 1.0))
    assert c["alpha"] == pytest.approx(sauter.alpha(0.5, 0.01))


def test_all_coefficients_finite_over_wide_parameter_space():
    """
    Nothing should produce a NaN anywhere in the parameter space the solver
    can reach, including the extreme collisionalities found at the edge.
    """
    ft = np.linspace(0.0, 0.99, 12)
    nu = np.logspace(-6, 4, 12)
    Z = np.array([1.0, 1.5, 3.0, 6.0])
    F, N, ZZ = np.meshgrid(ft, nu, Z, indexing="ij")

    for arr in (
        sauter.L31(F, N, ZZ),
        sauter.L32(F, N, ZZ),
        sauter.L34(F, N, ZZ),
        sauter.alpha(F, N),
        sauter.F33(sauter.ft_eff_33(F, N, ZZ), ZZ),
    ):
        assert np.all(np.isfinite(arr))
