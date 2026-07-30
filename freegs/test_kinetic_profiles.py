"""
Tests for KineticProfiles: species bookkeeping, derivatives, and the
smoothness and edge-integrity validation.
"""

import numpy as np
import pytest

from freegs.kinetic_profiles import (
    KineticProfiles,
    ProfileValidationError,
    ProfileWarning,
    mtanh_profile,
)
from freegs.sauter import ELEMENTARY_CHARGE


def make_profiles(**kwargs):
    """A well behaved H-mode-like set of profiles."""
    x = np.linspace(0.0, 1.0, 121)
    defaults = {
        "psi_n": x,
        "ne": mtanh_profile(x, core=6.0e19, ped=3.5e19, sep=6.0e18),
        "Te": mtanh_profile(x, core=3000.0, ped=700.0, sep=80.0),
        "Ti": mtanh_profile(x, core=2800.0, ped=700.0, sep=90.0),
        "Zeff": 1.8,
    }
    defaults.update(kwargs)
    return KineticProfiles(**defaults)


# ----------------------------------------------------------------------------
# mtanh helper
# ----------------------------------------------------------------------------


def test_mtanh_is_smooth_positive_and_bounded():
    x = np.linspace(0.0, 1.0, 2001)
    y = mtanh_profile(x, core=6e19, ped=3.5e19, sep=6e18)
    assert np.all(y > 0.0)
    assert y[0] == pytest.approx(6e19, rel=1e-3)
    # No spurious oscillation: monotonically decreasing
    assert np.all(np.diff(y) < 0.0)
    # Bounded second difference, i.e. no kinks
    d2 = np.diff(y, 2)
    assert np.max(np.abs(d2)) < 1e-2 * np.max(y)


def test_mtanh_rejects_nonpositive_separatrix():
    with pytest.raises(ValueError, match="positive separatrix"):
        mtanh_profile(np.linspace(0, 1, 10), 1e19, 5e18, 0.0)


# ----------------------------------------------------------------------------
# Species bookkeeping
# ----------------------------------------------------------------------------


def test_quasineutrality_and_Zeff():
    """
    ne = Zi ni + Zimp nimp  and  ne Zeff = Zi^2 ni + Zimp^2 nimp
    must both hold at every radius.
    """
    kin = make_profiles(Zeff=2.5, ion_Z=1.0, impurity_Z=6.0)
    x = np.linspace(0.0, 1.0, 41)

    ne, ni, nimp, Zeff = kin.ne(x), kin.ni(x), kin.nimp(x), kin.Zeff(x)

    assert np.allclose(ne, 1.0 * ni + 6.0 * nimp, rtol=1e-12)
    assert np.allclose(ne * Zeff, 1.0**2 * ni + 6.0**2 * nimp, rtol=1e-12)
    assert np.all(ni > 0.0)
    assert np.all(nimp > 0.0)


def test_pure_plasma_has_no_impurity():
    kin = make_profiles(Zeff=1.0, ion_Z=1.0, impurity_Z=6.0)
    x = np.linspace(0.0, 1.0, 21)
    assert np.allclose(kin.nimp(x), 0.0)
    assert np.allclose(kin.ni(x), kin.ne(x))


def test_n_ion_is_sum_of_ion_species():
    kin = make_profiles(Zeff=2.0)
    x = np.linspace(0.0, 1.0, 21)
    assert np.allclose(kin.n_ion(x), kin.ni(x) + kin.nimp(x), rtol=1e-12)


def test_Zeff_outside_range_is_rejected():
    with pytest.raises(ProfileValidationError, match="negative main ion"):
        make_profiles(Zeff=7.0, impurity_Z=6.0)
    with pytest.raises(ProfileValidationError, match="negative main ion"):
        make_profiles(Zeff=0.5, ion_Z=1.0)


def test_impurity_Z_must_exceed_ion_Z():
    with pytest.raises(ProfileValidationError, match="must exceed"):
        make_profiles(ion_Z=6.0, impurity_Z=6.0)


def test_Zeff_profile_accepted():
    x = np.linspace(0.0, 1.0, 121)
    Zeff = 1.5 + 1.0 * x**2
    kin = make_profiles(Zeff=Zeff)
    assert kin.Zeff(0.0) == pytest.approx(1.5, abs=1e-6)
    assert kin.Zeff(1.0) == pytest.approx(2.5, abs=1e-6)


# ----------------------------------------------------------------------------
# Pressures and derivatives
# ----------------------------------------------------------------------------


def test_pressure_composition():
    kin = make_profiles(Zeff=1.8)
    x = np.linspace(0.0, 1.0, 41)

    pe = ELEMENTARY_CHARGE * kin.ne(x) * kin.Te(x)
    pi = ELEMENTARY_CHARGE * kin.n_ion(x) * kin.Ti(x)

    assert np.allclose(kin.pe(x), pe, rtol=1e-12)
    assert np.allclose(kin.pi(x), pi, rtol=1e-12)
    assert np.allclose(kin.pressure(x), pe + pi, rtol=1e-12)
    assert np.allclose(kin.Rpe(x), pe / (pe + pi), rtol=1e-12)


def test_analytic_pressure_derivative_matches_finite_difference():
    """
    dpressure_dpsin is assembled analytically from the component splines, so
    check it against a central difference of pressure().
    """
    for Zeff in (1.0, 1.8, np.linspace(1.2, 3.0, 121)):
        kin = make_profiles(Zeff=Zeff)
        x = np.linspace(0.02, 0.98, 60)
        h = 1e-5
        fd = (kin.pressure(x + h) - kin.pressure(x - h)) / (2.0 * h)
        assert np.allclose(kin.dpressure_dpsin(x), fd, rtol=1e-5)


def test_log_gradients_match_finite_difference():
    kin = make_profiles()
    x = np.linspace(0.02, 0.98, 60)
    h = 1e-6
    for value, grad in (
        (kin.ne, kin.dlnne_dpsin),
        (kin.Te, kin.dlnTe_dpsin),
        (kin.Ti, kin.dlnTi_dpsin),
        (kin.pressure, kin.dlnp_dpsin),
    ):
        fd = (np.log(value(x + h)) - np.log(value(x - h))) / (2.0 * h)
        assert np.allclose(grad(x), fd, rtol=1e-4)


def test_gradients_are_negative_for_decreasing_profiles():
    kin = make_profiles()
    x = np.linspace(0.05, 0.99, 50)
    assert np.all(kin.dlnne_dpsin(x) < 0.0)
    assert np.all(kin.dlnTe_dpsin(x) < 0.0)
    assert np.all(kin.dlnp_dpsin(x) < 0.0)


# ----------------------------------------------------------------------------
# Positivity and extrapolation
# ----------------------------------------------------------------------------


def test_profiles_stay_positive_everywhere_including_beyond_the_boundary():
    """
    The log-spline representation makes negative density or temperature
    impossible, even where the solver evaluates slightly outside [0, 1].
    """
    kin = make_profiles()
    x = np.linspace(-0.1, 1.1, 400)
    assert np.all(kin.ne(x) > 0.0)
    assert np.all(kin.Te(x) > 0.0)
    assert np.all(kin.Ti(x) > 0.0)
    assert np.all(kin.pressure(x) > 0.0)


def test_extrapolation_is_C1():
    """
    Log-linear continuation past the last data point must be continuous in
    value and first derivative.
    """
    x = np.linspace(0.0, 0.9, 60)
    kin = KineticProfiles(
        x,
        mtanh_profile(x, 6e19, 3.5e19, 6e18),
        mtanh_profile(x, 3000.0, 700.0, 80.0),
        validate=False,
    )
    h = 1e-7
    for value, grad in ((kin.ne, kin.dlnne_dpsin), (kin.Te, kin.dlnTe_dpsin)):
        assert value(0.9 - h) == pytest.approx(value(0.9 + h), rel=1e-6)
        assert grad(0.9 - h) == pytest.approx(grad(0.9 + h), rel=1e-4)


def test_nonpositive_input_rejected():
    x = np.linspace(0.0, 1.0, 20)
    ne = np.linspace(5e19, 0.0, 20)  # hits zero at the edge
    with pytest.raises(ProfileValidationError, match="non-positive"):
        KineticProfiles(x, ne, np.full(20, 1000.0))


def test_malformed_grid_rejected():
    good = np.full(20, 1000.0)
    with pytest.raises(ProfileValidationError, match="strictly increasing"):
        KineticProfiles(np.linspace(1.0, 0.0, 20), np.full(20, 5e19), good)
    with pytest.raises(ProfileValidationError, match="at least 4 points"):
        KineticProfiles(np.linspace(0.0, 1.0, 3), np.full(3, 5e19), np.full(3, 1e3))
    with pytest.raises(ProfileValidationError, match="shape"):
        KineticProfiles(np.linspace(0.0, 1.0, 20), np.full(19, 5e19), good)
    x = np.linspace(0.0, 1.0, 20)
    bad = np.full(20, 5e19)
    bad[5] = np.nan
    with pytest.raises(ProfileValidationError, match="non-finite"):
        KineticProfiles(x, bad, good)


# ----------------------------------------------------------------------------
# Edge validation, which is the point of the exercise
# ----------------------------------------------------------------------------


def test_good_profiles_produce_no_warnings():
    kin = make_profiles()
    assert kin.messages == []


def test_warns_when_data_stops_short_of_the_boundary():
    x = np.linspace(0.0, 0.85, 60)
    with pytest.warns(ProfileWarning, match="extrapolated"):
        kin = KineticProfiles(
            x,
            mtanh_profile(x, 6e19, 3.5e19, 6e18),
            mtanh_profile(x, 3000.0, 700.0, 80.0),
        )
    assert any("extrapolated" in m for m in kin.messages)


def _loglinear(x, axis_value, edge_value):
    """
    A profile decaying exponentially from axis_value to edge_value.

    Log-linear, so it is perfectly represented by the log-spline and has a
    constant, well resolved logarithmic gradient. That isolates the edge
    magnitude checks from the smoothness and resolution checks.
    """
    return axis_value * (edge_value / axis_value) ** x


def test_warns_on_collapsing_edge_temperature():
    """A temperature that falls to almost nothing at the boundary."""
    x = np.linspace(0.0, 1.0, 121)
    with pytest.warns(ProfileWarning, match="below the floor"):
        kin = KineticProfiles(x, _loglinear(x, 6e19, 6e18), _loglinear(x, 3000.0, 0.2))
    assert any("T_e(1)" in m for m in kin.messages)


def test_warns_on_collapsing_edge_density():
    x = np.linspace(0.0, 1.0, 121)
    with pytest.warns(ProfileWarning, match="below the floor"):
        kin = KineticProfiles(
            x, _loglinear(x, 6e19, 1.0e14), _loglinear(x, 3000.0, 80.0)
        )
    assert any("n_e(1)" in m for m in kin.messages)


def test_no_edge_warning_for_healthy_pedestal():
    """
    The edge checks must not fire for a realistic H-mode pedestal, otherwise
    they are useless. This is the counterpart to the two tests above.
    """
    x = np.linspace(0.0, 1.0, 201)
    kin = KineticProfiles(
        x,
        mtanh_profile(x, 6e19, 3.5e19, 6e18),
        mtanh_profile(x, 3000.0, 700.0, 80.0),
        mtanh_profile(x, 2800.0, 700.0, 90.0),
        Zeff=1.8,
    )
    assert kin.messages == []


def test_warns_on_underresolved_edge():
    """A pedestal on a grid too coarse to represent it."""
    x = np.concatenate([np.linspace(0.0, 0.85, 18), [0.9, 1.0]])
    ne = mtanh_profile(x, 6e19, 3.5e19, 6e18, ped_width=0.02)
    Te = mtanh_profile(x, 3000.0, 700.0, 80.0, ped_width=0.02)
    with pytest.warns(ProfileWarning):
        kin = KineticProfiles(x, ne, Te)
    assert any("psi_n >= 0.9" in m or "under-resolved" in m for m in kin.messages)


def test_detects_spline_ringing_from_noisy_data():
    """
    Noisy data fitted with an interpolating spline rings; the validator must
    notice, and smoothing must fix it.
    """
    rng = np.random.default_rng(1234)
    x = np.linspace(0.0, 1.0, 121)
    clean_ne = mtanh_profile(x, 6e19, 3.5e19, 6e18)
    clean_Te = mtanh_profile(x, 3000.0, 700.0, 80.0)
    noise = 1.0 + 0.25 * rng.standard_normal(x.size)
    noisy_ne = np.abs(clean_ne * noise)
    noisy_Te = np.abs(clean_Te * (1.0 + 0.25 * rng.standard_normal(x.size)))

    with pytest.warns(ProfileWarning):
        rough = KineticProfiles(x, noisy_ne, noisy_Te)
    assert rough.messages

    # Smoothing the logarithm should tame it
    smoothed = KineticProfiles(x, noisy_ne, noisy_Te, smooth=3.0, validate=False)
    xd = np.linspace(0.0, 1.0, 401)

    # Far fewer sign changes in the gradient than the interpolating fit
    def sign_changes(kin):
        g = kin.dlnne_dpsin(xd)
        return int(np.count_nonzero(np.diff(np.sign(g)) != 0))

    assert sign_changes(smoothed) < sign_changes(rough)


def test_smoothing_options_all_work():
    x = np.linspace(0.0, 1.0, 121)
    ne = mtanh_profile(x, 6e19, 3.5e19, 6e18)
    Te = mtanh_profile(x, 3000.0, 700.0, 80.0)
    for smooth in (None, 1.0, "auto"):
        kin = KineticProfiles(x, ne, Te, smooth=smooth, validate=False)
        assert np.all(np.isfinite(kin.ne(np.linspace(0, 1, 50))))
        assert np.all(kin.ne(np.linspace(0, 1, 50)) > 0.0)

    with pytest.raises(ValueError, match="Unknown smooth"):
        KineticProfiles(x, ne, Te, smooth="nonsense", validate=False)


def test_Ti_defaults_to_Te():
    x = np.linspace(0.0, 1.0, 60)
    Te = mtanh_profile(x, 3000.0, 700.0, 80.0)
    kin = KineticProfiles(x, mtanh_profile(x, 6e19, 3.5e19, 6e18), Te)
    xs = np.linspace(0.0, 1.0, 20)
    assert np.allclose(kin.Ti(xs), kin.Te(xs))


def test_validate_can_be_deferred():
    x = np.linspace(0.0, 0.8, 40)
    kin = KineticProfiles(
        x,
        mtanh_profile(x, 6e19, 3.5e19, 6e18),
        mtanh_profile(x, 3000.0, 700.0, 80.0),
        validate=False,
    )
    assert kin.messages == []
    messages = kin.validate(warn=False)
    assert any("extrapolated" in m for m in messages)


def test_summary_and_report_run():
    kin = make_profiles()
    assert "psi_n" in kin.summary()
    kin.report()  # must not raise
