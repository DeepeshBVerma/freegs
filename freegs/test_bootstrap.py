"""
Tests for the bootstrap-consistent profile class and the current decomposition.

The internal-consistency checks here are the strongest statements available
without an external reference code: that <j.B> reconstructed from the
converged p' and ff' reproduces what the Sauter formulas asked for, that the
ohmic, bootstrap and diamagnetic currents sum to the target Ip, and that the
2D and radial routes to those currents agree.
"""

import numpy as np
import pytest

from freegs import control, geqdsk, jtor, machine, picard
from freegs.bootstrap import BootstrapProfiles, analyse_equilibrium, make_psin_grid
from freegs.equilibrium import Equilibrium
from freegs.gradshafranov import mu0
from freegs.kinetic_profiles import KineticProfiles, mtanh_profile


def make_kinetic(npoints=121, Zeff=1.8):
    x = np.linspace(0.0, 1.0, npoints)
    return KineticProfiles(
        x,
        mtanh_profile(x, core=6.0e19, ped=3.5e19, sep=6.0e18),
        mtanh_profile(x, core=3000.0, ped=700.0, sep=80.0),
        mtanh_profile(x, core=2800.0, ped=700.0, sep=90.0),
        Zeff=Zeff,
    )


def solve_diiid(Ip=-1.4e6, fvac=-3.231962138124, npsi=40, ntheta=96, nx=65, ny=65):
    """Solve a DIII-D equilibrium with a self-consistent bootstrap current."""
    eq = Equilibrium(
        tokamak=machine.DIIID(),
        Rmin=0.1,
        Rmax=2.8,
        Zmin=-1.8,
        Zmax=1.8,
        nx=nx,
        ny=ny,
    )
    profiles = BootstrapProfiles(
        eq, make_kinetic(), Ip=Ip, fvac=fvac, npsi=npsi, ntheta=ntheta
    )
    constrain = control.constrain(
        xpoints=[(1.285, -1.176), (1.2, 1.0)],
        gamma=1e-12,
        isoflux=[(1.285, -1.176, 1.2, 1.2)],
    )
    constrain(eq)
    picard.solve(eq, profiles, constrain, show=False, rtol=1e-5, maxits=100)
    return eq, profiles


@pytest.fixture(scope="module")
def diiid():
    return solve_diiid()


# ----------------------------------------------------------------------------
# The exact relation the whole scheme rests on
# ----------------------------------------------------------------------------


def test_jdotB_identity_holds_pointwise(diiid):
    """
    J.B = f p' + (f'/mu0) B^2 pointwise, in FreeGS conventions.

    This is the relation inverted to obtain ff' from <j.B>. Verify it directly
    on the (R, Z) grid using Maxwell's equations rather than the profile
    class: mu0 J_pol = f' B_pol follows from mu0 J = curl(B) with
    B_tor = f(psi)/R, and J_tor comes from the Grad-Shafranov equation.
    """
    eq, profiles = diiid
    R, Z = eq.R, eq.Z
    psi = eq.psi()
    psin = (psi - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)

    # Restrict to the well resolved interior of the core
    core = (eq.mask > 0.99) & (psin > 0.2) & (psin < 0.8)
    assert core.sum() > 100

    pprime = profiles.pprime(psin)
    ffprime = profiles.ffprime(psin)
    f = np.asarray(eq.fpol(psin))
    fprime = ffprime / f

    Br, Bz = eq.Br(R, Z), eq.Bz(R, Z)
    Bpol2 = Br**2 + Bz**2
    Btor = f / R
    B2 = Bpol2 + Btor**2

    # J_tor from Grad-Shafranov, J_pol from mu0 J_pol = f' B_pol
    Jtor = R * pprime + ffprime / (mu0 * R)
    JdotB = Jtor * Btor + (fprime / mu0) * Bpol2

    identity = f * pprime + (fprime / mu0) * B2

    scale = np.max(np.abs(identity[core]))
    assert np.max(np.abs(JdotB[core] - identity[core])) / scale < 1e-12


def test_jdotB_reconstruction_matches_sauter(diiid):
    """
    Going the other way: the converged ff' must reproduce the <j.B> that the
    Sauter formulas produced. This tests the inversion
    ff' = mu0 f (<j.B> - f p') / <B^2>.
    """
    _, profiles = diiid
    _, err = profiles.analysis().check_jdotB_consistency()
    assert np.max(np.abs(err)) < 1e-10


# ----------------------------------------------------------------------------
# Current decomposition
# ----------------------------------------------------------------------------


def test_components_sum_to_target_current(diiid):
    eq, profiles = diiid
    a = profiles.analysis()

    assert a.I_ohm + a.I_bs + a.I_dia == pytest.approx(a.Ip, rel=1e-12)
    assert a.Ip == pytest.approx(profiles.Ip, rel=1e-3)
    # And the equilibrium itself carries that current
    assert eq.plasmaCurrent() == pytest.approx(profiles.Ip, rel=1e-3)


def test_component_current_densities_sum_to_Jtor(diiid):
    """The 2D component arrays must add up to the Jtor that was solved for."""
    eq, profiles = diiid
    total = (
        profiles.Jtor_components["ohm"]
        + profiles.Jtor_components["bs"]
        + profiles.Jtor_components["dia"]
    )
    scale = np.max(np.abs(eq.Jtor))
    assert np.max(np.abs(total - eq.Jtor)) / scale < 1e-12


def test_radial_and_2d_integrals_agree(diiid):
    """
    Independent routes to the component currents: Romberg quadrature over the
    (R, Z) grid versus integration of dI/dpsi_n built from flux surface
    averages. Agreement validates the flux surface averaging.
    """
    _, profiles = diiid
    a = profiles.analysis()
    radial = a.currents_from_radial_integral()

    for key, direct in (("ohm", a.I_ohm), ("bs", a.I_bs), ("dia", a.I_dia)):
        assert radial[key] == pytest.approx(direct, rel=0.05), (
            f"{key}: radial {radial[key]:.4e} vs 2D {direct:.4e}"
        )


def test_bootstrap_is_co_current(diiid):
    """
    The bootstrap current must add to the plasma current for these outwardly
    decreasing profiles, and be a physically plausible fraction of it.
    """
    _, profiles = diiid
    a = profiles.analysis()

    assert np.sign(a.I_bs) == np.sign(a.Ip)
    assert np.sign(a.I_ohm) == np.sign(a.Ip)
    assert 0.0 < a.f_bootstrap < 0.5


def test_bootstrap_current_vanishes_on_axis(diiid):
    """
    No trapped particles on axis means no bootstrap current there, and the
    conductivity reduces to Spitzer.
    """
    _, profiles = diiid
    neo = profiles.neo

    assert neo["ft"][0] == 0.0
    assert neo["jdotB_bs"][0] == pytest.approx(0.0)
    assert neo["L31"][0] == pytest.approx(0.0)


def test_loop_voltage_sign_and_magnitude(diiid):
    """
    Vloop must drive current in the direction of Ip, and be of a plausible
    magnitude for a tokamak of this size and temperature.
    """
    _, profiles = diiid
    a = profiles.analysis()

    # <j.B>_ohm = sigma Vloop f <1/R^2> / 2pi must be co-current, i.e. the
    # resulting Jtor_ohm = f <j.B>_ohm / (R <B^2>) has the sign of Ip
    assert np.sign(a.I_ohm) == np.sign(a.Ip)
    assert 0.01 < abs(a.Vloop) < 20.0


def test_pressure_comes_from_the_kinetic_profiles(diiid):
    """
    p(psi) must be exactly the kinetic pressure, not an integral of p'.
    """
    _, profiles = diiid
    kin = profiles.kinetic
    x = np.linspace(0.0, 1.0, 25)
    assert np.allclose(profiles.pressure(x), kin.pressure(x), rtol=1e-12)

    # And p' must be its derivative with respect to psi
    dpsi = profiles.psi_bndry - profiles.psi_axis
    assert np.allclose(profiles.pprime(x), kin.dpressure_dpsin(x) / dpsi, rtol=1e-12)


def test_fpol_consistent_with_ffprime(diiid):
    """
    fpol integrates ff', so d(f^2/2)/dpsi must return ff'.
    """
    _, profiles = diiid
    dpsi = profiles.psi_bndry - profiles.psi_axis
    x = np.linspace(0.05, 0.95, 40)
    h = 1e-5
    f_hi = np.asarray(profiles.fpol(x + h))
    f_lo = np.asarray(profiles.fpol(x - h))
    # d(f^2/2)/dpsi_n / dpsi = ff'
    numeric = (f_hi**2 - f_lo**2) / (2.0 * 2.0 * h) / dpsi
    assert np.allclose(numeric, profiles.ffprime(x), rtol=1e-4)


def test_fpol_matches_fvac_at_boundary(diiid):
    _, profiles = diiid
    assert float(profiles.fpol(1.0)) == pytest.approx(profiles.fvac(), rel=1e-10)


def test_edge_cutoff_carries_little_current(diiid):
    """
    The frozen-geometry region beyond psi_max must be a small correction, not
    a dominant contribution, otherwise the cutoff is doing the physics.
    """
    _, profiles = diiid
    a = profiles.analysis()

    frozen = ~a.traced
    frozen[0] = False
    I_frozen = np.trapezoid(a.dIdpsin["bs"][frozen], a.psi_n[frozen])
    assert abs(I_frozen) < 0.1 * abs(a.I_bs)


def test_geometry_frozen_beyond_psi_max(diiid):
    """Geometric inputs must be constant above the cutoff, by construction."""
    _, profiles = diiid
    neo = profiles.neo
    beyond = profiles.psi_n > profiles.psi_max

    for key in ("ft", "eps", "q", "Rgeo", "B2_avg", "R2inv_avg"):
        vals = neo[key][beyond]
        assert np.allclose(vals, vals[0]), f"{key} is not frozen beyond psi_max"


def test_kinetic_gradients_still_vary_beyond_psi_max(diiid):
    """
    Freezing the geometry must not freeze the profiles: the local pedestal
    gradient has to keep driving bootstrap current above the cutoff.
    """
    _, profiles = diiid
    neo = profiles.neo
    beyond = profiles.psi_n >= profiles.psi_max
    assert np.count_nonzero(beyond) >= 3
    assert not np.allclose(neo["dlnp_dpsi"][beyond], neo["dlnp_dpsi"][beyond][0])


# ----------------------------------------------------------------------------
# Sign robustness: the physics must not depend on FreeGS sign conventions
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("Ip", "fvac"),
    [
        (-1.4e6, -3.231962138124),
        (1.4e6, 3.231962138124),
        (-1.4e6, 3.231962138124),
    ],
)
def test_bootstrap_co_current_for_all_sign_conventions(Ip, fvac):
    """
    Whatever the signs of the plasma current and toroidal field, the bootstrap
    current must add to Ip. This is the check that pins down the sign
    convention discussed in sauter.py.
    """
    _, profiles = solve_diiid(Ip=Ip, fvac=fvac, npsi=32, ntheta=64, nx=65, ny=65)
    a = profiles.analysis()

    assert np.sign(a.I_bs) == np.sign(Ip), (
        f"Ip={Ip:.2e}, fvac={fvac:.2f}: I_bs={a.I_bs:.3e} is counter-current"
    )
    assert np.sign(a.I_ohm) == np.sign(Ip)
    assert a.Ip == pytest.approx(Ip, rel=1e-3)
    assert 0.0 < a.f_bootstrap < 0.5


# ----------------------------------------------------------------------------
# Physical trends
# ----------------------------------------------------------------------------


def test_bootstrap_fraction_grows_with_pressure():
    """
    f_bs should rise with beta_p. Doubling the temperature at fixed Ip roughly
    doubles the pressure gradient driving the bootstrap current, while the
    ohmic current is constrained to make up the remainder.
    """
    x = np.linspace(0.0, 1.0, 121)
    ne = mtanh_profile(x, 6.0e19, 3.5e19, 6.0e18)

    fractions = []
    for scale in (0.6, 1.4):
        kin = KineticProfiles(
            x,
            ne,
            mtanh_profile(x, 3000.0 * scale, 700.0 * scale, 80.0 * scale),
            mtanh_profile(x, 2800.0 * scale, 700.0 * scale, 90.0 * scale),
            Zeff=1.8,
        )
        eq = Equilibrium(
            tokamak=machine.DIIID(),
            Rmin=0.1,
            Rmax=2.8,
            Zmin=-1.8,
            Zmax=1.8,
            nx=65,
            ny=65,
        )
        profiles = BootstrapProfiles(
            eq, kin, Ip=-1.4e6, fvac=-3.231962138124, npsi=32, ntheta=64
        )
        constrain = control.constrain(
            xpoints=[(1.285, -1.176), (1.2, 1.0)],
            gamma=1e-12,
            isoflux=[(1.285, -1.176, 1.2, 1.2)],
        )
        constrain(eq)
        picard.solve(eq, profiles, constrain, show=False, rtol=1e-5, maxits=100)
        fractions.append(profiles.analysis().f_bootstrap)

    assert fractions[1] > fractions[0], (
        f"f_bs did not increase with pressure: {fractions}"
    )


# ----------------------------------------------------------------------------
# Construction and error handling
# ----------------------------------------------------------------------------


def test_constructor_validates_arguments():
    eq = Equilibrium(
        tokamak=machine.DIIID(), Rmin=0.1, Rmax=2.8, Zmin=-1.8, Zmax=1.8, nx=33, ny=33
    )
    kin = make_kinetic()
    with pytest.raises(ValueError, match="npsi"):
        BootstrapProfiles(eq, kin, Ip=1e6, fvac=2.0, npsi=4)
    with pytest.raises(ValueError, match="psi_min"):
        BootstrapProfiles(eq, kin, Ip=1e6, fvac=2.0, psi_min=0.5, psi_max=0.4)
    with pytest.raises(ValueError, match="psi_min"):
        BootstrapProfiles(eq, kin, Ip=1e6, fvac=2.0, psi_max=1.5)


def test_analysis_before_solve_raises():
    eq = Equilibrium(
        tokamak=machine.DIIID(), Rmin=0.1, Rmax=2.8, Zmin=-1.8, Zmax=1.8, nx=33, ny=33
    )
    profiles = BootstrapProfiles(eq, make_kinetic(), Ip=1e6, fvac=2.0)
    with pytest.raises(RuntimeError, match="No solution yet"):
        profiles.analysis()
    with pytest.raises(RuntimeError, match="Jtor has been called"):
        profiles.pprime(0.5)


def test_fpol_before_solve_is_fvac():
    eq = Equilibrium(
        tokamak=machine.DIIID(), Rmin=0.1, Rmax=2.8, Zmin=-1.8, Zmax=1.8, nx=33, ny=33
    )
    profiles = BootstrapProfiles(eq, make_kinetic(), Ip=1e6, fvac=2.5)
    assert np.allclose(profiles.fpol(np.linspace(0, 1, 5)), 2.5)
    assert np.allclose(profiles.ffprime(np.linspace(0, 1, 5)), 0.0)


def test_summary_and_report_run(diiid):
    _, profiles = diiid
    a = profiles.analysis()
    text = a.summary()
    assert "Bootstrap fraction" in text
    assert "Loop voltage" in text
    a.report()  # must not raise


# ----------------------------------------------------------------------------
# Diagnostic decomposition of a fixed equilibrium
# ----------------------------------------------------------------------------


def test_diagnostic_recovers_self_consistent_solution(diiid):
    """
    Running analyse_equilibrium on a bootstrap-consistent equilibrium, with the
    profiles that generated it, must recover the same decomposition.

    This is the round trip: the self-consistent solve goes from <j.B> to ff',
    and the diagnostic goes from ff' back to <j.B>.
    """
    eq, profiles = diiid
    sc = profiles.analysis()
    diag = analyse_equilibrium(eq, profiles.kinetic, npsi=40, ntheta=96)

    assert diag.f_bootstrap == pytest.approx(sc.f_bootstrap, rel=0.02)
    assert diag.I_bs == pytest.approx(sc.I_bs, rel=0.02)
    assert diag.I_ohm == pytest.approx(sc.I_ohm, rel=0.02)


def test_diagnostic_implied_loop_voltage_is_constant(diiid):
    """
    The strongest available check on the decomposition. For a self-consistent
    solution the ohmic current was generated from a single loop voltage, so the
    Vloop inferred radially from <j.B>_ohm = sigma_neo <E.B> must come out
    radially constant and equal to it.
    """
    eq, profiles = diiid
    sc = profiles.analysis()
    diag = analyse_equilibrium(eq, profiles.kinetic, npsi=40, ntheta=96)

    v = diag.Vloop_implied[diag.traced]
    v = v[np.isfinite(v)]
    assert len(v) > 10

    assert np.median(v) == pytest.approx(sc.Vloop, rel=1e-3)
    spread = (np.max(v) - np.min(v)) / abs(np.median(v))
    assert spread < 0.02, f"Implied Vloop varies by {100 * spread:.2f}% across radius"


def test_diagnostic_pressure_mismatch_is_zero_for_consistent_profiles(diiid):
    eq, profiles = diiid
    diag = analyse_equilibrium(eq, profiles.kinetic, npsi=40, ntheta=96)
    assert diag.pressure_mismatch < 1e-10


def test_diagnostic_detects_inconsistent_profiles(diiid):
    """
    Supplying profiles that do not describe the equilibrium must be flagged,
    since the bootstrap estimate is only as good as that agreement.
    """
    eq, _ = diiid
    x = np.linspace(0.0, 1.0, 121)
    wrong = KineticProfiles(
        x,
        mtanh_profile(x, core=3.0e19, ped=1.5e19, sep=3.0e18),
        mtanh_profile(x, core=1200.0, ped=300.0, sep=40.0),
        Zeff=1.8,
    )
    diag = analyse_equilibrium(eq, wrong, npsi=32, ntheta=64)
    assert diag.pressure_mismatch > 0.1
    assert "WARNING" in diag.summary()


def test_diagnostic_current_integral_matches_equilibrium(diiid):
    """
    The radial integral of dI/dpsi_n must reproduce the equilibrium's own
    plasma current, which validates the flux surface quadrature against the
    2D Romberg integration FreeGS uses.
    """
    eq, profiles = diiid
    diag = analyse_equilibrium(eq, profiles.kinetic, npsi=64, ntheta=128)
    assert diag.Ip == pytest.approx(diag.Ip_equilibrium, rel=0.01)


def test_diagnostic_components_sum_to_total(diiid):
    eq, profiles = diiid
    diag = analyse_equilibrium(eq, profiles.kinetic, npsi=32, ntheta=64)
    assert diag.I_ohm + diag.I_bs + diag.I_dia == pytest.approx(diag.Ip, rel=1e-12)
    assert np.allclose(
        diag.dIdpsin["total"],
        diag.dIdpsin["ohm"] + diag.dIdpsin["bs"] + diag.dIdpsin["dia"],
    )


def test_diagnostic_works_on_a_prescribed_profile_equilibrium():
    """
    The diagnostic must work on any equilibrium, not just one produced by
    BootstrapProfiles. Here the current profile comes from the usual shape
    function, so the bootstrap current is genuinely an added estimate and the
    inferred Vloop has no reason to be radially constant.
    """
    eq = Equilibrium(
        tokamak=machine.DIIID(),
        Rmin=0.1,
        Rmax=2.8,
        Zmin=-1.8,
        Zmax=1.8,
        nx=65,
        ny=65,
    )
    kin = make_kinetic()
    profiles = jtor.ConstrainPaxisIp(
        eq, float(kin.pressure(0.0)), -1.4e6, -3.231962138124
    )
    constrain = control.constrain(
        xpoints=[(1.285, -1.176), (1.2, 1.0)],
        gamma=1e-12,
        isoflux=[(1.285, -1.176, 1.2, 1.2)],
    )
    constrain(eq)
    picard.solve(eq, profiles, constrain, show=False, rtol=1e-5, maxits=100)

    diag = analyse_equilibrium(eq, kin, npsi=32, ntheta=64)

    assert np.sign(diag.I_bs) == np.sign(eq.plasmaCurrent())
    assert 0.0 < diag.f_bootstrap < 0.5
    assert diag.Ip == pytest.approx(diag.Ip_equilibrium, rel=0.02)
    diag.report()  # must not raise


def test_geqdsk_read_accepts_freeqdsk_dataclass(diiid, tmp_path):
    """
    Regression test for the limiter check in geqdsk.read.

    freeqdsk >= 0.5 returns a GEQDSKFile dataclass whose __getitem__ takes
    attribute names, so the old `if "rlim" in data` fell back to the iteration
    protocol and raised TypeError before any equilibrium could be read. This
    blocks reading any G-EQDSK file, which is the route external validation
    data comes in by.

    The reconstruction itself is a nonlinear solve and may not converge for an
    arbitrary file, so this only asserts that the limiter is parsed and the
    wall set, not that the solve succeeds.
    """
    eq, _ = diiid
    path = tmp_path / "test.geqdsk"
    with open(path, "w") as fh:
        geqdsk.write(eq, fh)

    tokamak = machine.DIIID()
    tokamak.wall = None

    with open(path) as fh:
        try:
            geqdsk.read(fh, tokamak, show=False, maxits=1)
        except TypeError as exc:  # pragma: no cover - the bug being guarded
            pytest.fail(f"geqdsk.read failed parsing the limiter: {exc}")
        except RuntimeError:
            # Nonlinear solve did not converge in one iteration, which is
            # expected and irrelevant here
            pass

    assert tokamak.wall is not None, "limiter was not read from the G-EQDSK file"
    assert len(tokamak.wall.R) > 3


def test_make_psin_grid():
    psi_n, traced = make_psin_grid(32, 0.02, 0.995)
    assert psi_n[0] == 0.0
    assert psi_n[-1] == pytest.approx(1.0)
    assert np.all(np.diff(psi_n) > 0.0)
    assert psi_n[traced][0] == pytest.approx(0.02)
    assert psi_n[traced][-1] == pytest.approx(0.995)
    assert traced.stop - traced.start == 32

    with pytest.raises(ValueError, match="npsi"):
        make_psin_grid(4, 0.02, 0.995)
    with pytest.raises(ValueError, match="psi_min"):
        make_psin_grid(32, 0.5, 0.4)
