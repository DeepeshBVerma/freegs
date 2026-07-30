#!/usr/bin/env python
"""
DIII-D equilibrium with a self-consistent neoclassical bootstrap current.

The current profile is not prescribed here. Instead the electron and ion
density and temperature profiles fix the pressure, the Sauter neoclassical
formulas [Sauter, Angioni and Lin-Liu, Phys. Plasmas 6, 2834 (1999)] give the
flux surface averaged parallel current, and ff'(psi) follows from

    ff' = mu0 f ( <j.B> - f p' ) / <B^2>

The loop voltage is adjusted at every Picard iteration so that the total
plasma current matches the requested Ip. After the solve, the ohmic and
bootstrap contributions are separated for analysis.

Compare with 16-DIIID.py, which uses the same machine with a prescribed
current profile shape.
"""

import matplotlib.pyplot as plt
import numpy as np

import freegs

#########################################
# Kinetic profiles
#
# H-mode-like profiles with a pedestal. The mtanh form is smooth, strictly
# positive and has bounded gradients at psi_n = 1, so it does not break down
# in the edge region. Use 201 points so the pedestal is properly resolved:
# with too coarse a grid, KineticProfiles will warn that the edge gradient
# driving the bootstrap current is set by the spline rather than the data.

psi_n = np.linspace(0.0, 1.0, 201)

ne = freegs.mtanh_profile(
    psi_n, core=6.0e19, ped=3.5e19, sep=6.0e18
)  # Electron density [m^-3]
Te = freegs.mtanh_profile(
    psi_n, core=3000.0, ped=700.0, sep=80.0
)  # Electron temperature [eV]
Ti = freegs.mtanh_profile(
    psi_n, core=2800.0, ped=700.0, sep=90.0
)  # Ion temperature [eV]

kinetic = freegs.KineticProfiles(
    psi_n,
    ne,
    Te,
    Ti,
    Zeff=1.8,  # Effective charge; may also be an array
    ion_Z=1.0,
    ion_A=2.0,  # Deuterium main ion
    impurity_Z=6.0,  # Carbon carries Z_eff
)

# Inspect the profiles, including the edge diagnostics. Any concern about
# smoothness or behaviour as psi_n -> 1 is listed here.
kinetic.report()
print()

#########################################
# Create the machine and equilibrium domain, as in 16-DIIID.py

tokamak = freegs.machine.DIIID()

eq = freegs.Equilibrium(
    tokamak=tokamak,
    Rmin=0.1,
    Rmax=2.8,  # Radial domain
    Zmin=-1.8,
    Zmax=1.8,  # Height range
    nx=129,
    ny=129,
)  # Number of grid points

#########################################
# Bootstrap-consistent plasma profiles

profiles = freegs.BootstrapProfiles(
    eq,
    kinetic,
    Ip=-1533632,  # Target plasma current [Amps]
    fvac=-3.231962138124,  # Vacuum f = R*Bt
    npsi=64,  # Flux surfaces for the neoclassical calculation
    ntheta=128,  # Poloidal points per flux surface
    psi_max=0.995,  # Geometry frozen beyond this normalised flux
)

#########################################
# Coil current constraints, as in 16-DIIID.py

xpoints = [
    (1.285, -1.176),  # (R,Z) locations of X-points
    (1.2, 1.0),
]

isoflux = [(1.285, -1.176, 1.2, 1.2)]  # (R1,Z1, R2,Z2) pair of locations

constrain = freegs.control.constrain(xpoints=xpoints, gamma=1e-12, isoflux=isoflux)

constrain(eq)

#########################################
# Nonlinear solve
#
# This is slower than a prescribed-profile solve: flux surfaces are traced and
# the neoclassical coefficients recomputed at every iteration.

freegs.solve(eq,
             profiles,
             constrain,
             show=True,
             rtol=1e-5,
             maxits=100
)

print("Done!")
print()
print(f"Plasma current:       {eq.plasmaCurrent():e} Amps")
print(f"Pressure on axis:     {eq.pressure(0.0):e} Pascals")
print(f"Pressure at boundary: {eq.pressure(1.0):e} Pascals")
print(f"Plasma poloidal beta: {eq.poloidalBeta():e}")
print(f"Plasma volume:        {eq.plasmaVolume():e} m^3")
print()

eq.tokamak.printCurrents()
print()

#########################################
# Separate the ohmic and bootstrap contributions

analysis = profiles.analysis()
analysis.report()

##############################################
# Plots

# Equilibrium
axis = eq.plot(show=False)
tokamak.plot(axis=axis, show=False)
constrain.plot(axis=axis, show=True)

# Kinetic profiles and their edge behaviour
kinetic.plot()

# Current decomposition and neoclassical coefficients
analysis.plot()

# Radial current profile, split by contribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

ax[0].plot(analysis.psi_n, analysis.dIdpsin["total"], "k", lw=2, label="total")
ax[0].plot(analysis.psi_n, analysis.dIdpsin["ohm"], label="ohmic")
ax[0].plot(analysis.psi_n, analysis.dIdpsin["bs"], label="bootstrap")
ax[0].plot(analysis.psi_n, analysis.dIdpsin["dia"], label="diamagnetic")
ax[0].axvline(profiles.psi_max, color="r", ls=":", lw=1, label="geometry cutoff")
ax[0].set_xlabel(r"Normalised $\psi$")
ax[0].set_ylabel(r"$dI/d\psi_N$ [A]")
ax[0].set_title(f"Bootstrap fraction {analysis.f_bootstrap:.3f}")
ax[0].legend()
ax[0].grid(alpha=0.3)

# Safety factor. Note this profile is set by the neoclassical conductivity
# rather than chosen: sigma_neo ~ Te^(3/2) makes the ohmic current strongly
# peaked, which drives q on axis down.
psinorm, q = eq.q()
ax[1].plot(psinorm, q)
ax[1].set_xlabel(r"Normalised $\psi$")
ax[1].set_ylabel("Safety factor")
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

##############################################
# Save to geqdsk file

from freegs import geqdsk

with open("diiid-bootstrap.geqdsk", "w") as f:
    geqdsk.write(eq, f)
