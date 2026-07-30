#!/usr/bin/env python
"""
Compare the current profile and q profile of a bootstrap-consistent DIII-D
equilibrium against a prescribed-profile one.

Runs the same machine, domain, coil constraints, Ip and fvac twice, changing
only the profile object:

  A. freegs.jtor.ConstrainPaxisIp  - current profile shape prescribed
                                     (as in 16-DIIID.py)
  B. freegs.BootstrapProfiles      - current profile from the Sauter
                                     neoclassical parallel current
                                     (as in 17-DIIID-bootstrap.py)

Both equilibria are then diagnosed through the *same* flux surface machinery,
so the comparison is not contaminated by two different post-processing paths.
Four quantities are compared:

  q(psi_N)                     safety factor
  dI/dpsi_N                    radial distribution of toroidal current [A]
  <Jtor>(psi_N)                flux surface averaged current density [A/m^2]
  Jtor(R) on the midplane      the same current in real space [A/m^2]

For run B the current is additionally split into its ohmic, bootstrap and
diamagnetic parts, which sum exactly to the total.


CHOOSING THE BASELINE
---------------------

The two profile classes cannot be matched on the whole pressure profile:
ConstrainPaxisIp uses (1 - psi_N^alpha_m)^alpha_n, which vanishes at the
boundary, while p = n*T carries a pedestal. So you must pick what to match,
and the choice changes the answer:

  MATCH = "betap"   match poloidal beta, i.e. the stored energy. Recommended.
                    Same global confinement, so differences in q and in the
                    current profile are mostly attributable to the current
                    drive mechanism.

  MATCH = "paxis"   match the on-axis pressure. Looks natural but is
                    misleading: for these profiles it leaves beta_p differing
                    by ~2.9x, and that mismatch dominates every difference in
                    the comparison.

Neither isolates the bootstrap current perfectly, because the pressure
*shapes* still differ. For that you would re-solve with ff'_bs removed and
everything else held fixed.


NOTE ON THE SIGN OF q
---------------------

|q| is plotted, not q. The base class freegs.jtor.Profile.fpol returns
sqrt(2*val + fvac**2), which drops the sign of fvac, so for the DIII-D
fvac = -3.232 run A reports f = +3.275 and therefore a positive q, while run B
(which preserves the sign) reports a negative q. The magnitudes are directly
comparable; the signs are not. See the discussion in the module docstring of
freegs/sauter.py for why the sign matters.
"""

import matplotlib.pyplot as plt
import numpy as np

import freegs
from freegs.bootstrap import analyse_equilibrium, make_psin_grid
from freegs.fluxsurface import FluxSurfaces
from freegs.gradshafranov import mu0

# ============================================================================
# Configuration
# ============================================================================

IP = -1.4e6  # Target plasma current [Amps]
FVAC = -3.231962138124  # Vacuum f = R*Bt [T m]

MATCH = "betap"  # "betap" (recommended) or "paxis"

NX, NY = 129, 129  # Equilibrium grid
NPSI, NTHETA = 64, 128  # Flux surface resolution for diagnostics
RTOL, MAXITS = 1e-5, 100  # Picard tolerance and iteration cap

SAVE_FIGURE = "compare-bootstrap.png"  # Set to None to skip saving

# Colours, chosen to stay distinguishable in greyscale
C_PRESCRIBED = "#5B6B7A"
C_BOOTSTRAP = "#1F6F6B"
C_BS_PART = "#C08A2E"
C_OHM_PART = "#7E8A95"


def kinetic_profiles():
    """H-mode-like profiles with a resolved pedestal, as in 17-DIIID-bootstrap.py."""
    psi_n = np.linspace(0.0, 1.0, 201)
    return freegs.KineticProfiles(
        psi_n,
        freegs.mtanh_profile(psi_n, core=6.0e19, ped=3.5e19, sep=6.0e18),
        freegs.mtanh_profile(psi_n, core=3000.0, ped=700.0, sep=80.0),
        freegs.mtanh_profile(psi_n, core=2800.0, ped=700.0, sep=90.0),
        Zeff=1.8,
        ion_Z=1.0,
        ion_A=2.0,
        impurity_Z=6.0,
    )


def new_equilibrium():
    return freegs.Equilibrium(
        tokamak=freegs.machine.DIIID(),
        Rmin=0.1,
        Rmax=2.8,
        Zmin=-1.8,
        Zmax=1.8,
        nx=NX,
        ny=NY,
    )


def apply_constraints(eq):
    constrain = freegs.control.constrain(
        xpoints=[(1.285, -1.176), (1.2, 1.0)],
        gamma=1e-12,
        isoflux=[(1.285, -1.176, 1.2, 1.2)],
    )
    constrain(eq)
    return constrain


# ============================================================================
# Diagnostics, applied identically to both equilibria
# ============================================================================


def current_and_q_profiles(eq, npsi=NPSI, ntheta=NTHETA):
    """
    Current and safety factor profiles from traced flux surfaces.

    Works for any solved Equilibrium, whatever produced its profiles, so the
    two runs are measured with the same ruler.

    p' and ff' are flux functions, so on a surface

        <Jtor>     = p' <R> + ff' <1/R> / mu0
        <Jtor / R> = p' + ff' <1/R^2> / mu0
        dI/dpsi    = (dV/dpsi / 2pi) <Jtor / R>

    Returns a dict of 1D arrays on the returned psi_n grid.
    """
    psi_n, traced_slice = make_psin_grid(npsi, 0.02, 0.995)
    x = psi_n[traced_slice]

    fs = FluxSurfaces(eq, x, ntheta=ntheta)

    pprime = np.asarray(eq.pprime(x), dtype=float)
    ffprime = np.asarray(eq.ffprime(x), dtype=float)
    dpsi = eq.psi_bndry - eq.psi_axis

    Rinv_avg = fs.average(1.0 / fs.R)

    Jtor_avg = pprime * fs.R_avg + ffprime * Rinv_avg / mu0
    JtorOverR_avg = pprime + ffprime * fs.R2inv_avg / mu0
    dIdpsin = (fs.dVdpsi / (2.0 * np.pi)) * JtorOverR_avg * dpsi

    # Integrate over the whole of [0, 1], not just the traced range, or the
    # comparison against eq.plasmaCurrent() is short by a few percent purely
    # because the innermost 2% and outermost 0.5% of the flux are missing.
    # dI/dpsi_N is smooth on axis, and although dV/dpsi diverges
    # logarithmically at the separatrix the region is narrow, so a linear
    # continuation to each end is adequate for a consistency check.
    x_full = np.concatenate(([0.0], x, [1.0]))
    dI_full = np.concatenate(
        (
            [dIdpsin[0] + (dIdpsin[1] - dIdpsin[0]) * (0.0 - x[0]) / (x[1] - x[0])],
            dIdpsin,
            [
                dIdpsin[-1]
                + (dIdpsin[-1] - dIdpsin[-2]) * (1.0 - x[-1]) / (x[-1] - x[-2])
            ],
        )
    )

    return {
        "psi_n": x,
        "q": fs.q,
        "Jtor_avg": Jtor_avg,
        "dIdpsin": dIdpsin,
        "Ip_from_profile": float(np.trapezoid(dI_full, x_full)),
        "Ip_traced_only": float(np.trapezoid(dIdpsin, x)),
    }


def midplane_current(eq, npoints=400):
    """Jtor along the horizontal line through the magnetic axis."""
    Zaxis = eq.Zmagnetic()
    R = np.linspace(eq.Rmin, eq.Rmax, npoints)
    Z = np.full_like(R, Zaxis)

    psi = eq.psiRZ(R, Z)
    psi_n = (psi - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)

    inside = (psi_n >= 0.0) & (psi_n <= 1.0)
    pn = np.clip(psi_n, 0.0, 1.0)

    Jtor = R * np.asarray(eq.pprime(pn)) + np.asarray(eq.ffprime(pn)) / (mu0 * R)
    return R, np.where(inside, Jtor, np.nan)


def q_at(prof, xq):
    """|q| at given psi_N, interpolated from the traced grid."""
    return np.interp(xq, prof["psi_n"], np.abs(prof["q"]))


# ============================================================================
# Solves
# ============================================================================


def solve_bootstrap(kinetic):
    print("=" * 74)
    print("Run B: bootstrap-consistent (BootstrapProfiles)")
    print("=" * 74)
    eq = new_equilibrium()
    profiles = freegs.BootstrapProfiles(
        eq, kinetic, Ip=IP, fvac=FVAC, npsi=NPSI, ntheta=NTHETA
    )
    constrain = apply_constraints(eq)
    freegs.solve(eq, profiles, constrain, show=False, rtol=RTOL, maxits=MAXITS)
    print(f"  Ip      = {eq.plasmaCurrent():.6e} A")
    print(f"  beta_p  = {eq.poloidalBeta():.6f}")
    print(f"  p_axis  = {float(eq.pressure(0.0)):.1f} Pa")
    print()
    return eq, profiles


def solve_prescribed(betap_target, paxis_target):
    if MATCH == "betap":
        label = f"ConstrainBetapIp, beta_p = {betap_target:.5f}"
        maker = lambda eq: freegs.jtor.ConstrainBetapIp(eq, betap_target, IP, FVAC)
    elif MATCH == "paxis":
        label = f"ConstrainPaxisIp, p_axis = {paxis_target:.1f} Pa"
        maker = lambda eq: freegs.jtor.ConstrainPaxisIp(eq, paxis_target, IP, FVAC)
    else:
        raise ValueError(f"MATCH must be 'betap' or 'paxis', got {MATCH!r}")

    print("=" * 74)
    print(f"Run A: prescribed profile shape ({label})")
    print("=" * 74)
    eq = new_equilibrium()
    profiles = maker(eq)
    constrain = apply_constraints(eq)
    freegs.solve(eq, profiles, constrain, show=False, rtol=RTOL, maxits=MAXITS)
    print(f"  Ip      = {eq.plasmaCurrent():.6e} A")
    print(f"  beta_p  = {eq.poloidalBeta():.6f}")
    print(f"  p_axis  = {float(eq.pressure(0.0)):.1f} Pa")
    print()
    return eq, profiles


# ============================================================================
# Report and plot
# ============================================================================


def report(eqA, eqB, profB, kinetic):
    pA = current_and_q_profiles(eqA)
    pB = current_and_q_profiles(eqB)

    anB = profB.analysis()
    anA = analyse_equilibrium(eqA, kinetic, npsi=NPSI, ntheta=NTHETA)

    print("=" * 74)
    print("Comparison")
    print("=" * 74)
    print(
        f"{'quantity':<28s} {'prescribed':>14s} {'bootstrap':>14s} {'ratio B/A':>11s}"
    )

    def row(name, a, b, fmt="{:10.4f}"):
        r = b / a if a != 0 else np.nan
        print(f"{name:<28s} {fmt.format(a):>14s} {fmt.format(b):>14s} {r:>11.3f}")

    row("Ip [A]", eqA.plasmaCurrent(), eqB.plasmaCurrent(), "{:12.5e}")
    row("beta_p", eqA.poloidalBeta(), eqB.poloidalBeta())
    row("p_axis [Pa]", float(eqA.pressure(0.0)), float(eqB.pressure(0.0)), "{:10.1f}")
    print()
    row("|q| on axis (psi_N=0.02)", q_at(pA, 0.02), q_at(pB, 0.02))
    row("|q| at psi_N=0.5", q_at(pA, 0.5), q_at(pB, 0.5))
    row("|q| at psi_N=0.95", q_at(pA, 0.95), q_at(pB, 0.95))
    row("min |q|", np.min(np.abs(pA["q"])), np.min(np.abs(pB["q"])))
    print()
    row(
        "peak |<Jtor>| [A/m^2]",
        np.max(np.abs(pA["Jtor_avg"])),
        np.max(np.abs(pB["Jtor_avg"])),
        "{:12.4e}",
    )
    row(
        "|<Jtor>| at psi_N=0.02",
        abs(pA["Jtor_avg"][0]),
        abs(pB["Jtor_avg"][0]),
        "{:12.4e}",
    )
    row("l_i", eqA.internalInductance(), eqB.internalInductance())
    print()
    print(f"  bootstrap fraction, run A (inferred)      f_bs = {anA.f_bootstrap:.4f}")
    print(f"  bootstrap fraction, run B (self-consist.) f_bs = {anB.f_bootstrap:.4f}")
    print(
        f"  run B  I_ohm / I_bs / I_dia = {anB.I_ohm:.4e} / "
        f"{anB.I_bs:.4e} / {anB.I_dia:.4e} A"
    )
    print(f"  run B  Vloop = {anB.Vloop:.4f} V")
    vA = anA.Vloop_implied[anA.traced]
    vA = vA[np.isfinite(vA)]
    print(
        f"  run A  implied Vloop: median {np.median(vA):+.4f} V, "
        f"range [{np.min(vA):+.3f}, {np.max(vA):+.3f}] V"
    )
    print()
    print("  Consistency of the radial current integral against eq.plasmaCurrent():")
    for tag, prof, eq in (("A", pA, eqA), ("B", pB, eqB)):
        rel = abs(prof["Ip_from_profile"] - eq.plasmaCurrent()) / abs(
            eq.plasmaCurrent()
        )
        frac = abs(prof["Ip_traced_only"] / prof["Ip_from_profile"])
        print(
            f"    run {tag}: {prof['Ip_from_profile']:.4e} A vs "
            f"{eq.plasmaCurrent():.4e} A  (rel. diff {rel:.2e}; "
            f"traced range holds {100 * frac:.1f}%)"
        )
    print()

    print("  Interpretation")
    print("  " + "-" * 70)
    pax_ratio = float(eqB.pressure(0.0)) / float(eqA.pressure(0.0))
    print(
        f"    Matching {MATCH} leaves p_axis differing by {pax_ratio:.2f}x. The two\n"
        "    pressure shapes cannot both be matched: the shape function puts all its\n"
        "    pressure in the core, while n*T carries a pedestal. Switch MATCH to see\n"
        "    the mirror image of this."
    )
    dq = q_at(pB, 0.5) / q_at(pA, 0.5) - 1.0
    dli = eqB.internalInductance() / eqA.internalInductance() - 1.0
    print(
        f"    Mid-radius |q| differs by {100 * dq:+.0f}% and l_i by {100 * dli:+.0f}%,\n"
        "    so at matched stored energy the neoclassical current profile is genuinely\n"
        "    broader than the prescribed one, not just differently normalised."
    )
    print()

    return pA, pB, anB


def plot(pA, pB, anB, eqA, eqB):
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.5))

    # --- Safety factor
    a = ax[0, 0]
    a.plot(pA["psi_n"], np.abs(pA["q"]), color=C_PRESCRIBED, lw=1.8, label="prescribed")
    a.plot(pB["psi_n"], np.abs(pB["q"]), color=C_BOOTSTRAP, lw=1.8, label="bootstrap")
    a.axhline(1.0, color="0.6", ls=":", lw=1)
    a.set_ylabel(r"$|q|$")
    a.set_title("Safety factor")
    a.legend()

    # --- Radial current distribution
    a = ax[0, 1]
    a.plot(
        pA["psi_n"],
        pA["dIdpsin"],
        color=C_PRESCRIBED,
        lw=1.8,
        label="prescribed, total",
    )
    a.plot(
        pB["psi_n"], pB["dIdpsin"], color=C_BOOTSTRAP, lw=1.8, label="bootstrap, total"
    )
    a.plot(
        anB.psi_n,
        anB.dIdpsin["ohm"],
        color=C_OHM_PART,
        lw=1.3,
        ls="--",
        label="bootstrap run: ohmic",
    )
    a.plot(
        anB.psi_n,
        anB.dIdpsin["bs"],
        color=C_BS_PART,
        lw=1.6,
        ls="--",
        label="bootstrap run: BS part",
    )
    a.plot(
        anB.psi_n,
        anB.dIdpsin["dia"],
        color="#9C6B9C",
        lw=1.0,
        ls=":",
        label="bootstrap run: diamagnetic",
    )
    a.set_ylabel(r"$dI/d\psi_N$  [A]")
    a.set_title("Radial current distribution")
    a.legend(fontsize="small")

    # --- Flux surface averaged current density
    a = ax[1, 0]
    a.plot(pA["psi_n"], pA["Jtor_avg"], color=C_PRESCRIBED, lw=1.8, label="prescribed")
    a.plot(pB["psi_n"], pB["Jtor_avg"], color=C_BOOTSTRAP, lw=1.8, label="bootstrap")
    a.set_ylabel(r"$\langle J_\phi \rangle$  [A m$^{-2}$]")
    a.set_title("Flux surface averaged current density")
    a.legend()

    # --- Midplane cut
    a = ax[1, 1]
    RA, JA = midplane_current(eqA)
    RB, JB = midplane_current(eqB)
    a.plot(RA, JA, color=C_PRESCRIBED, lw=1.8, label="prescribed")
    a.plot(RB, JB, color=C_BOOTSTRAP, lw=1.8, label="bootstrap")
    a.axvline(eqA.Rmagnetic(), color=C_PRESCRIBED, ls=":", lw=1)
    a.axvline(eqB.Rmagnetic(), color=C_BOOTSTRAP, ls=":", lw=1)
    a.set_xlabel("$R$  [m]")
    a.set_ylabel(r"$J_\phi$  [A m$^{-2}$]")
    a.set_title("Current density on the midplane (dotted: magnetic axis)")
    a.legend()

    for a in ax.flat:
        a.grid(alpha=0.3)
        if a.get_xlabel() == "":
            a.set_xlabel(r"Normalised $\psi$")

    fig.suptitle(
        f"DIII-D: prescribed vs bootstrap-consistent current profile "
        f"(matched on {MATCH}, $I_p$ = {IP / 1e6:.2f} MA)",
        fontsize=12,
    )
    fig.tight_layout()

    if SAVE_FIGURE:
        fig.savefig(SAVE_FIGURE, dpi=150)
        print(f"Figure written to {SAVE_FIGURE}")

    plt.show()


def main():
    kinetic = kinetic_profiles()
    if kinetic.messages:
        print("Kinetic profile warnings:")
        for m in kinetic.messages:
            print(f"  - {m}")
        print()

    # Run B first: its beta_p is what run A is matched to
    eqB, profB = solve_bootstrap(kinetic)
    eqA, _ = solve_prescribed(
        betap_target=eqB.poloidalBeta(),
        paxis_target=float(eqB.pressure(0.0)),
    )

    pA, pB, anB = report(eqA, eqB, profB, kinetic)
    plot(pA, pB, anB, eqA, eqB)


if __name__ == "__main__":
    main()
