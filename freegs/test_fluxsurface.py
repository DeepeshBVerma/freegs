"""
Tests for flux surface geometry, flux surface averages and trapped fraction.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from freegs import control, jtor, machine, picard
from freegs.equilibrium import Equilibrium
from freegs.fluxsurface import _G, FluxSurfaces, trapped_fraction

# ----------------------------------------------------------------------------
# Trapped fraction, checked against analytic limits
# ----------------------------------------------------------------------------


def test_G_function_limits():
    """G(x) = 2 - 3 sqrt(1-x) + (1-x)^(3/2); G(0)=0, G(1)=2, G ~ 3x^2/4."""
    assert _G(0.0) == pytest.approx(0.0)
    assert _G(1.0) == pytest.approx(2.0)
    x = 1.0e-3
    assert _G(x) == pytest.approx(0.75 * x**2, rel=1e-3)
    # Monotonically increasing
    xs = np.linspace(0.0, 1.0, 50)
    assert np.all(np.diff(_G(xs)) > 0.0)


def test_trapped_fraction_uniform_field():
    """A uniform |B| has no trapped particles."""
    B = 2.0
    ft = trapped_fraction(B2_avg=B**2, B_avg=B, Bmax=B, G_over_B2_avg=_G(1.0) / B**2)
    assert ft == pytest.approx(0.0, abs=1e-12)


def _circular_field(eps, ntheta):
    """
    |B| around a large aspect ratio circular flux surface, B = B0/(1 + eps cos t),
    together with the poloidal angle grid and Bmax.

    In this limit the poloidal angle is the correct flux surface average
    measure, so averages are plain integrals over t.
    """
    t = np.linspace(0.0, 2.0 * np.pi, ntheta)
    return t, 1.0 / (1.0 + eps * np.cos(t)), 1.0 / (1.0 - eps)


def _circular_ft_approx(eps, ntheta=20001):
    """Lin-Liu & Miller trapped fraction for a circular surface."""
    t, B, Bmax = _circular_field(eps, ntheta)

    def avg(x):
        return np.trapezoid(x, t) / (2.0 * np.pi)

    return trapped_fraction(
        B2_avg=avg(B**2),
        B_avg=avg(B),
        Bmax=Bmax,
        G_over_B2_avg=avg(_G(B / Bmax) / B**2),
    )


def _circular_ft_exact(eps, ntheta=200001):
    """
    Trapped fraction from the exact definition, Eq. (12) of Sauter et al.,

        f_t = 1 - (3/4) <B^2> int_0^(1/Bmax) lambda dlambda / <sqrt(1 - lambda B)>

    evaluated by direct quadrature for a circular surface.
    """
    t, B, Bmax = _circular_field(eps, ntheta)

    def avg(x):
        return np.trapezoid(x, t) / (2.0 * np.pi)

    B2 = avg(B**2)

    def integrand(lam):
        return lam / avg(np.sqrt(np.maximum(1.0 - lam * B, 0.0)))

    val, _ = quad(integrand, 0.0, 1.0 / Bmax, limit=200)
    return 1.0 - 0.75 * B2 * val


@pytest.mark.parametrize("eps", [0.002, 0.005, 0.01])
def test_trapped_fraction_small_epsilon_asymptote(eps):
    """
    In the large aspect ratio circular limit the exact trapped fraction is
    f_t -> 1.46 sqrt(eps). The Lin-Liu & Miller combination used here is
    built to reproduce that coefficient.
    """
    ft = _circular_ft_approx(eps)
    assert ft / np.sqrt(eps) == pytest.approx(1.46, abs=0.03)


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2, 0.3, 0.5])
def test_trapped_fraction_matches_exact_definition(eps):
    """
    Compare the approximation against a direct quadrature of Eq. (12) over a
    range of aspect ratios. Lin-Liu & Miller quote a few percent accuracy.
    """
    approx = _circular_ft_approx(eps)
    exact = _circular_ft_exact(eps)
    assert approx == pytest.approx(exact, rel=0.05), (
        f"eps={eps}: approx {approx:.4f} vs exact {exact:.4f}"
    )


def test_trapped_fraction_bounds():
    ft = trapped_fraction(4.0, 2.0, 3.0, 0.4)
    assert 0.0 <= ft <= 1.0


# ----------------------------------------------------------------------------
# Flux surface tracing against a solved equilibrium
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solved_eq():
    """A simple diverted equilibrium to trace surfaces on."""
    eq = Equilibrium(
        tokamak=machine.TestTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=129,
        ny=129,
    )
    profiles = jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    constrain = control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)], isoflux=[(1.1, -0.6, 1.1, 0.6)]
    )
    picard.solve(eq, profiles, constrain, show=False)
    return eq


def test_safety_factor_matches_freegs(solved_eq):
    """
    The safety factor computed from the traced surfaces must agree with
    FreeGS's own independent calculation in critical.find_safety.
    """
    psin = np.linspace(0.1, 0.9, 9)
    fs = FluxSurfaces(solved_eq, psin, ntheta=256, nray=400)
    q_ref = np.asarray(solved_eq.q(psin))
    assert np.allclose(fs.q, q_ref, rtol=2e-3)


def test_volume_from_dVdpsi(solved_eq):
    """
    Integrating dV/dpsi over psi must reproduce the plasma volume. This
    validates both the flux surface tracing and the dl / |Bpol| weighting that
    all the flux surface averages use.
    """
    eq = solved_eq
    psin = np.linspace(1e-4, 0.99, 600)
    fs = FluxSurfaces(eq, psin, ntheta=256, nray=500)

    dpsi = abs(eq.psi_bndry - eq.psi_axis)
    volume = np.trapezoid(fs.dVdpsi * dpsi, psin)

    # Reference: direct grid integration inside the same psi_n limit
    R, Z = eq.R, eq.Z
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    psin_grid = (eq.psi() - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)
    sel = (eq.mask > 0.5) & (psin_grid < 0.99)
    reference = np.sum(2.0 * np.pi * R[sel]) * dR * dZ

    assert volume == pytest.approx(reference, rel=0.02)


def test_flux_surface_average_of_constant(solved_eq):
    """<c> = c for any constant."""
    fs = FluxSurfaces(solved_eq, np.linspace(0.1, 0.9, 5), ntheta=128)
    ones = np.ones_like(fs.R)
    assert np.allclose(fs.average(3.7 * ones), 3.7)


def test_B2_avg_consistency(solved_eq):
    """<B^2> must equal <Bpol^2> + f^2 <1/R^2>."""
    fs = FluxSurfaces(solved_eq, np.linspace(0.1, 0.9, 9), ntheta=256)
    assert np.allclose(fs.B2_avg, fs.Bpol2_avg + fs.fpol**2 * fs.R2inv_avg, rtol=1e-10)


def test_averages_ordering(solved_eq):
    """
    <B>^2 <= <B^2> by Cauchy-Schwarz, and <B> <= Bmax.
    """
    fs = FluxSurfaces(solved_eq, np.linspace(0.1, 0.95, 12), ntheta=256)
    assert np.all(fs.B_avg**2 <= fs.B2_avg * (1.0 + 1e-12))
    assert np.all(fs.B_avg <= fs.Bmax)
    assert np.all(fs.Bmin <= fs.B_avg)


def test_epsilon_increases_outward(solved_eq):
    fs = FluxSurfaces(solved_eq, np.linspace(0.05, 0.95, 20), ntheta=256)
    assert np.all(np.diff(fs.eps) > 0.0)
    assert np.all(fs.eps > 0.0)
    assert np.all(fs.eps < 1.0)


def test_trapped_fraction_increases_outward(solved_eq):
    fs = FluxSurfaces(solved_eq, np.linspace(0.05, 0.95, 20), ntheta=256)
    assert np.all(np.diff(fs.ft) > 0.0)
    assert np.all(fs.ft > 0.0)
    assert np.all(fs.ft < 1.0)


def test_traced_points_lie_on_the_requested_surface(solved_eq):
    """
    Evaluate psi at the traced points and confirm the normalised flux is what
    was asked for. This is the primary check on the ray casting and Newton
    refinement.
    """
    eq = solved_eq
    psin = np.linspace(0.05, 0.95, 10)
    fs = FluxSurfaces(eq, psin, ntheta=128, nray=300)

    psi_at_points = eq.psiRZ(fs.R, fs.Z)
    psin_at_points = (psi_at_points - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)

    for i, target in enumerate(psin):
        assert np.allclose(psin_at_points[i, :], target, atol=1e-6), (
            f"surface psi_n={target} deviates by "
            f"{np.max(np.abs(psin_at_points[i, :] - target)):.2e}"
        )


def test_check_reports_no_problems_for_good_equilibrium(solved_eq):
    fs = FluxSurfaces(solved_eq, np.linspace(0.05, 0.95, 20), ntheta=256)
    assert fs.check() == []


def test_average_rejects_wrong_shape(solved_eq):
    fs = FluxSurfaces(solved_eq, np.linspace(0.1, 0.9, 5), ntheta=128)
    with pytest.raises(ValueError, match="Expected array of shape"):
        fs.average(np.ones((3, 3)))
