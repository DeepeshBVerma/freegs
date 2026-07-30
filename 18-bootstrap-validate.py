#!/usr/bin/env python
"""
Validate the Sauter bootstrap implementation against an external case.

Fill in the three paths in the CONFIGURATION block below and run. The script
will then:

  1. Read the equilibrium from a G-EQDSK file.
  2. Read n_e, T_e, T_i (and optionally Z_eff) from an ASCII table.
  3. Check the supplied profiles are smooth and well behaved at the edge.
  4. Check the kinetic pressure reproduces the equilibrium's pressure.
  5. Compute the bootstrap current from the g-file geometry without
     re-solving, and compare the bootstrap fraction against your reference.
  6. Optionally re-solve the equilibrium self-consistently with the bootstrap
     current and compare.

Step 5 is the meaningful comparison against another code (TRANSP, ASTRA, NEO,
CHEASE, ...): it uses their equilibrium and their profiles, so any difference
is attributable to the neoclassical model rather than to a different
equilibrium. Step 6 answers a different question, namely what equilibrium
these profiles would produce on their own.

Note on the ohmic term in step 5: everything in the equilibrium's current that
is not bootstrap or diamagnetic is attributed to "ohmic". If the reference
case has neutral beam or RF current drive, that will appear there too, so
compare I_bs rather than I_ohm.

Note on runtime: step 1 is usually the slowest part. freegs.geqdsk.read does
not simply parse the file, it reconstructs the equilibrium with a nonlinear
solve and fits coil currents to the given boundary at every iteration. For a
machine described with shaped coils this can take several minutes, and for
some files it does not converge within maxits at all. That cost is in FreeGS's
reader, not in the bootstrap calculation, which needs only a solved
Equilibrium. If the read is the bottleneck, or fails to converge, consider
raising maxits, relaxing rtol, or passing a machine with a simpler coilset.


PROFILE FILE FORMAT
-------------------

Whitespace-separated ASCII, one row per radial point, increasing in the first
column. Lines beginning with # are ignored. Columns:

    psi_n    ne[m^-3]    Te[eV]    Ti[eV]    [Zeff]

The fifth column is optional; if absent, ZEFF below is used. For example:

    # psi_n     ne          Te      Ti     Zeff
    0.000    6.00e19     3000.0  2800.0   1.8
    0.050    5.95e19     2930.0  2740.0   1.8
    ...
    1.000    6.00e18       85.0    95.0   1.9

If your data is on sqrt(psi_n), rho_tor or r/a instead, convert it to psi_n
before writing the file: this code works in normalised poloidal flux
throughout. If your temperatures are in keV, set TEMPERATURE_IN_KEV = True.
"""

import numpy as np

import freegs
from freegs import geqdsk
from freegs.bootstrap import analyse_equilibrium

# ============================================================================
# CONFIGURATION -- edit this block
# ============================================================================

#: Path to the G-EQDSK equilibrium file.
GEQDSK_PATH = "PUT_THE_PATH_TO_YOUR_GEQDSK_FILE_HERE"

#: Path to the ASCII profile table described above.
PROFILE_PATH = "PUT_THE_PATH_TO_YOUR_PROFILE_FILE_HERE"

#: Reference bootstrap fraction I_bs/Ip to compare against, or None.
REFERENCE_F_BOOTSTRAP = None

#: Reference bootstrap current in Amps, or None. Use whichever of this and
#: REFERENCE_F_BOOTSTRAP your reference actually quotes.
REFERENCE_I_BOOTSTRAP = None

#: Set True if the profile file has temperatures in keV rather than eV.
TEMPERATURE_IN_KEV = False

#: Fallback Z_eff, used if the profile file has no fifth column.
ZEFF = 1.8

#: Main ion and impurity species.
ION_Z, ION_A = 1.0, 2.0  # Deuterium
IMPURITY_Z = 6.0  # Carbon

#: Spline smoothing of the logarithm of each profile. None interpolates the
#: data exactly, which rings on noisy experimental data. Try 1.0 to 10.0, or
#: "auto" for generalised cross-validation, if the smoothness check complains.
SMOOTH = None

#: Machine to use when reading the g-file. The coilset only needs to be able
#: to reproduce the given boundary; freegs.machine.DIIID() is a reasonable
#: default for DIII-D data.
MACHINE = freegs.machine.DIIID()

#: Set True to also run the self-consistent re-solve (step 6). Slower.
RUN_SELF_CONSISTENT = False

#: Numerical resolution of the neoclassical calculation.
NPSI, NTHETA, PSI_MAX = 64, 128, 0.995

# ============================================================================


def load_profiles(path):
    """Read the ASCII profile table and build a KineticProfiles."""
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(
            f"{path}: expected at least 4 columns "
            "(psi_n, ne, Te, Ti), got shape {data.shape}"
        )

    psi_n = data[:, 0]
    ne = data[:, 1]
    Te = data[:, 2]
    Ti = data[:, 3]
    Zeff = data[:, 4] if data.shape[1] >= 5 else ZEFF

    if TEMPERATURE_IN_KEV:
        Te = Te * 1.0e3
        Ti = Ti * 1.0e3

    print(f"Read {len(psi_n)} radial points from {path}")
    print(
        f"  psi_n in [{psi_n[0]:.4f}, {psi_n[-1]:.4f}], "
        f"ne(0) = {ne[0]:.3e} m^-3, Te(0) = {Te[0]:.1f} eV"
    )
    print()

    return freegs.KineticProfiles(
        psi_n,
        ne,
        Te,
        Ti,
        Zeff=Zeff,
        ion_Z=ION_Z,
        ion_A=ION_A,
        impurity_Z=IMPURITY_Z,
        smooth=SMOOTH,
    )


def main():
    for name, path in (("GEQDSK_PATH", GEQDSK_PATH), ("PROFILE_PATH", PROFILE_PATH)):
        if path.startswith("PUT_THE_PATH"):
            raise SystemExit(
                f"{name} has not been set. Edit the CONFIGURATION block at the "
                f"top of {__file__}, or see the module docstring for the "
                "expected profile file format."
            )

    # --- 1, 2, 3: load the equilibrium and the profiles
    print("=" * 76)
    print("1. Reading equilibrium")
    print("=" * 76)
    with open(GEQDSK_PATH) as fh:
        eq = geqdsk.read(fh, MACHINE, show=False)

    print(f"  Ip           = {eq.plasmaCurrent():.4e} A")
    print(f"  fvac         = {eq.fvac():.6f} T m")
    print(f"  psi_axis     = {eq.psi_axis:.6f}, psi_bndry = {eq.psi_bndry:.6f}")
    print(f"  R_magnetic   = {eq.Rmagnetic():.4f} m")
    print(f"  minor radius = {eq.minorRadius():.4f} m")
    print(f"  poloidal beta= {eq.poloidalBeta():.4f}")
    print()

    print("=" * 76)
    print("2, 3. Reading and checking profiles")
    print("=" * 76)
    kinetic = load_profiles(PROFILE_PATH)
    kinetic.report()
    print()

    # --- 4, 5: diagnostic decomposition on the given equilibrium
    print("=" * 76)
    print("4, 5. Bootstrap current from the given equilibrium (no re-solve)")
    print("=" * 76)
    analysis = analyse_equilibrium(
        eq, kinetic, npsi=NPSI, ntheta=NTHETA, psi_max=PSI_MAX
    )
    analysis.report()
    print()

    if REFERENCE_F_BOOTSTRAP is not None:
        err = analysis.f_bootstrap - REFERENCE_F_BOOTSTRAP
        print(
            f"  Reference f_bs = {REFERENCE_F_BOOTSTRAP:.4f}, "
            f"computed {analysis.f_bootstrap:.4f}, "
            f"difference {err:+.4f} "
            f"({100 * err / REFERENCE_F_BOOTSTRAP:+.1f}%)"
        )
    if REFERENCE_I_BOOTSTRAP is not None:
        err = analysis.I_bs - REFERENCE_I_BOOTSTRAP
        print(
            f"  Reference I_bs = {REFERENCE_I_BOOTSTRAP:.4e} A, "
            f"computed {analysis.I_bs:.4e} A, "
            f"difference {err:+.4e} A "
            f"({100 * err / abs(REFERENCE_I_BOOTSTRAP):+.1f}%)"
        )
    if REFERENCE_F_BOOTSTRAP is not None or REFERENCE_I_BOOTSTRAP is not None:
        print()

    analysis.plot()

    # --- 6: optional self-consistent re-solve
    if RUN_SELF_CONSISTENT:
        print("=" * 76)
        print("6. Self-consistent re-solve with the bootstrap current")
        print("=" * 76)

        eq2 = freegs.Equilibrium(
            tokamak=MACHINE,
            Rmin=eq.Rmin,
            Rmax=eq.Rmax,
            Zmin=eq.Zmin,
            Zmax=eq.Zmax,
            nx=eq.R.shape[0],
            ny=eq.R.shape[1],
        )
        profiles = freegs.BootstrapProfiles(
            eq2,
            kinetic,
            Ip=eq.plasmaCurrent(),
            fvac=eq.fvac(),
            npsi=NPSI,
            ntheta=NTHETA,
            psi_max=PSI_MAX,
        )

        # Constrain the new solve to the original separatrix shape
        sep = eq.separatrix(npoints=64)
        isoflux = [
            (sep[i, 0], sep[i, 1], sep[0, 0], sep[0, 1]) for i in range(1, len(sep))
        ]
        constrain = freegs.control.constrain(isoflux=isoflux, gamma=1e-12)
        constrain(eq2)

        freegs.solve(eq2, profiles, constrain, show=False, rtol=1e-5, maxits=100)

        selfcons = profiles.analysis()
        selfcons.report()
        print()
        print("  Comparison with the diagnostic decomposition:")
        print(
            f"    f_bs   diagnostic {analysis.f_bootstrap:.4f}   "
            f"self-consistent {selfcons.f_bootstrap:.4f}"
        )
        print(
            f"    I_bs   diagnostic {analysis.I_bs:.4e} A   "
            f"self-consistent {selfcons.I_bs:.4e} A"
        )
        print(
            "    A large difference means the given equilibrium's current "
            "profile is not the one these profiles would sustain inductively, "
            "which usually indicates auxiliary current drive or a "
            "non-relaxed profile."
        )
        selfcons.plot()


if __name__ == "__main__":
    main()
