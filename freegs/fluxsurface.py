"""
Flux surface geometry and flux surface averages.

Traces closed flux surfaces of an Equilibrium and evaluates the geometric
quantities needed by neoclassical transport formulas: flux surface averages
<B^2>, <B>, <1/R^2>, <R>, the safety factor, inverse aspect ratio, and the
trapped particle fraction.

The flux surface average of a quantity X is

    <X> = ( int X dl / |Bpol| ) / ( int dl / |Bpol| )

where the integrals are around the closed poloidal contour.

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

import warnings

import numpy as np
from scipy import interpolate

from . import critical


def trapped_fraction(B2_avg, B_avg, Bmax, G_over_B2_avg):
    """
    Trapped particle fraction using the Lin-Liu & Miller approximation.

    The exact definition, Eq. (12) of Sauter et al. (1999), is

        f_t = 1 - (3/4) <B^2> int_0^(1/Bmax) lambda dlambda / <sqrt(1 - lambda B)>

    Lin-Liu & Miller, Phys. Plasmas 2, 1666 (1995) bound the integral from
    above and below by applying Jensen's inequality and the Cauchy-Schwarz
    inequality respectively to <sqrt(1 - lambda B)>, giving

        f_t_upper = 1 - (<B^2> / (2 <B>^2)) G(<B>/Bmax)
        f_t_lower = 1 - (<B^2> / 2) < G(B/Bmax) / B^2 >

    with

        G(x) = 2 - 3 sqrt(1-x) + (1-x)^(3/2)

    and combine them as f_t = 0.75 f_t_upper + 0.25 f_t_lower. In the large
    aspect ratio circular limit the two bounds give 1.50 sqrt(eps) and
    1.35 sqrt(eps), and the weighted combination reproduces the exact
    1.46 sqrt(eps).

    Parameters
    ----------
    B2_avg:
        <B^2> [T^2]
    B_avg:
        <B> [T]
    Bmax:
        Maximum |B| on the flux surface [T]
    G_over_B2_avg:
        < G(B/Bmax) / B^2 > [T^-2]

    Returns
    -------
    Trapped fraction, clipped to [0, 1]
    """
    ft_upper = 1.0 - (B2_avg / (2.0 * B_avg**2)) * _G(B_avg / Bmax)
    ft_lower = 1.0 - (B2_avg / 2.0) * G_over_B2_avg

    return np.clip(0.75 * ft_upper + 0.25 * ft_lower, 0.0, 1.0)


def _G(x):
    """G(x) = 2 - 3 sqrt(1-x) + (1-x)^(3/2), for 0 <= x <= 1.

    Behaves as 3x^2/4 for small x, so no special handling is needed near
    x = 0 (uniform-|B| limit gives G(1) = 2).
    """
    one_minus = np.clip(1.0 - x, 0.0, 1.0)
    s = np.sqrt(one_minus)
    return 2.0 - 3.0 * s + one_minus * s


class FluxSurfaces:
    """
    Flux surface geometry for a set of normalised psi values.

    Attributes, all 1D arrays of length ``len(psinorm)`` unless noted:

    psinorm:
        Normalised poloidal flux of each surface (0 = axis, 1 = boundary)
    R, Z:
        (npsi, ntheta) arrays of surface coordinates [m]
    B2_avg:
        <B^2> [T^2]
    B_avg:
        <B> [T]
    Bmax, Bmin:
        Extremes of |B| on the surface [T]
    R2inv_avg:
        <1/R^2> [m^-2]
    R_avg:
        <R> [m]
    Bpol2_avg:
        <Bpol^2> = <|grad psi|^2 / R^2> [T^2]
    q:
        Safety factor (signed, following f and the plasma current)
    dVdpsi:
        dV/dpsi = 2 pi int dl / |Bpol| [m^3 Wb^-1]
    Rmax, Rmin:
        Extremes of R on the surface [m]
    Rgeo:
        (Rmax + Rmin) / 2 [m]
    eps:
        Inverse aspect ratio (Rmax - Rmin) / (Rmax + Rmin)
    ft:
        Trapped particle fraction
    """

    def __init__(
        self,
        eq,
        psinorm,
        psi=None,
        ntheta=128,
        nray=256,
        newton_iterations=3,
        psi_axis=None,
        psi_bndry=None,
        opoint=None,
        xpoint=None,
    ):
        """
        Parameters
        ----------
        eq:
            Equilibrium object. Used for Br, Bz, fpol and the domain extent.
        psinorm:
            1D array of normalised psi values in (0, 1) to trace.
        psi:
            Total psi on the (R, Z) grid. Defaults to eq.psi().
        ntheta:
            Number of poloidal points per surface.
        nray:
            Number of samples along each ray used for the initial bracket.
        newton_iterations:
            Newton refinement steps applied to each traced point.
        psi_axis, psi_bndry:
            Flux at the magnetic axis and plasma boundary. Taken from eq if
            not given. Supplying them explicitly lets this work for limited
            and fixed-boundary plasmas, where the boundary is not an X-point.
        opoint, xpoint:
            Lists of O- and X-points as returned by critical.find_critical.
            Computed if not given; passing them in avoids repeating that
            search on every Picard iteration.
        """
        self.eq = eq
        self.psinorm = np.atleast_1d(np.asarray(psinorm, dtype=float))

        if psi is None:
            psi = eq.psi()

        if psi_axis is None:
            psi_axis = eq.psi_axis
        if psi_bndry is None:
            psi_bndry = eq.psi_bndry
        if psi_bndry is None:
            raise ValueError("FluxSurfaces needs a plasma boundary flux")

        self.psi_axis = psi_axis
        self.psi_bndry = psi_bndry

        if opoint is None:
            opoint, xpoint = critical.find_critical(eq.R, eq.Z, psi)
            if not opoint:
                raise ValueError("No O-points found, cannot trace flux surfaces")

        self.Raxis, self.Zaxis = opoint[0][0:2]
        self._xpoint = xpoint

        # Normalised psi on the grid, and a spline for tracing
        psinorm_grid = (psi - psi_axis) / (psi_bndry - psi_axis)
        self._psinorm_func = interpolate.RectBivariateSpline(
            eq.R[:, 0], eq.Z[0, :], psinorm_grid
        )

        self._trace(ntheta, nray, newton_iterations)
        self._averages()

    # ------------------------------------------------------------------
    # Surface tracing
    # ------------------------------------------------------------------

    def _trace(self, ntheta, nray, newton_iterations):
        """
        Locate surface points by casting rays outward from the magnetic axis.

        Rays are equally spaced in geometric poloidal angle about the O-point.
        Along each ray psinorm increases monotonically outward, so the surface
        location is found by inverse interpolation followed by a few Newton
        steps against the psinorm spline.
        """
        eq = self.eq
        r0, z0 = self.Raxis, self.Zaxis

        theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)

        # Nudge the grid off any X-point direction, following critical.py
        xpt = self._xpoint
        if xpt:
            xtheta = np.arctan2(xpt[0][0] - r0, xpt[0][1] - z0) % (2.0 * np.pi)
            dtheta = theta[1] - theta[0]
            if np.any(np.abs(theta - xtheta) < 1.0e-3):
                theta = theta + 0.5 * dtheta

        self.theta = theta

        # Unit ray directions in (R, Z). Matching critical.py, theta is
        # measured from the +Z axis towards +R.
        dR = np.sin(theta)
        dZ = np.cos(theta)

        # Longest ray length that stays inside the domain, per direction
        with np.errstate(divide="ignore", invalid="ignore"):
            tR = np.where(
                dR > 0,
                (eq.Rmax - r0) / dR,
                np.where(dR < 0, (eq.Rmin - r0) / dR, np.inf),
            )
            tZ = np.where(
                dZ > 0,
                (eq.Zmax - z0) / dZ,
                np.where(dZ < 0, (eq.Zmin - z0) / dZ, np.inf),
            )
        # Small safety margin so splines are never evaluated outside the grid
        length = 0.999 * np.minimum(tR, tZ)

        # Sample psinorm along every ray: shape (ntheta, nray)
        s = np.linspace(0.0, 1.0, nray)
        Rray = r0 + np.outer(length * dR, s)
        Zray = z0 + np.outer(length * dZ, s)
        pn = self._psinorm_func.ev(Rray, Zray)

        # Enforce monotonicity so inverse interpolation is well posed. Beyond
        # the boundary psinorm can turn over (private flux region), and near
        # the axis spline noise can produce tiny non-monotonic wiggles.
        pn = np.maximum.accumulate(pn, axis=1)
        pn[:, 0] = 0.0

        npsi = len(self.psinorm)
        svals = np.empty((npsi, ntheta))
        for j in range(ntheta):
            svals[:, j] = np.interp(self.psinorm, pn[j, :], s)

        # Newton refinement: solve psinorm(R(s), Z(s)) = target along each ray
        target = self.psinorm[:, None]
        for _ in range(newton_iterations):
            R = r0 + svals * (length * dR)[None, :]
            Z = z0 + svals * (length * dZ)[None, :]
            val = self._psinorm_func.ev(R, Z)
            # d/ds = grad(psinorm) . (ray direction) * length
            dpdR = self._psinorm_func.ev(R, Z, dx=1)
            dpdZ = self._psinorm_func.ev(R, Z, dy=1)
            deriv = (dpdR * dR[None, :] + dpdZ * dZ[None, :]) * length[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                step = np.where(np.abs(deriv) > 0, (val - target) / deriv, 0.0)
            svals = np.clip(svals - step, 0.0, 1.0)

        self.R = r0 + svals * (length * dR)[None, :]
        self.Z = z0 + svals * (length * dZ)[None, :]

    # ------------------------------------------------------------------
    # Flux surface averages
    # ------------------------------------------------------------------

    def _averages(self):
        """Evaluate the flux surface averages on the traced surfaces."""
        eq = self.eq
        R, Z = self.R, self.Z

        Br = eq.Br(R, Z)
        Bz = eq.Bz(R, Z)
        Bpol = np.sqrt(Br**2 + Bz**2)

        # f = R*Btor from the current profile iterate.
        fpol = np.asarray(eq.fpol(self.psinorm)).reshape(-1, 1)
        self.fpol = fpol[:, 0]
        Btor = fpol / R
        B = np.sqrt(Bpol**2 + Btor**2)

        # Arc length from central differences of position w.r.t. index, so the
        # sum over the closed contour is the trapezoidal rule.
        dr_di = (np.roll(R, -1, axis=1) - np.roll(R, 1, axis=1)) / 2.0
        dz_di = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / 2.0
        dl = np.sqrt(dr_di**2 + dz_di**2)

        # Guard against a vanishing poloidal field (magnetic axis, X-point)
        Bpol_safe = np.maximum(Bpol, 1.0e-10 * np.max(B))
        w = dl / Bpol_safe
        norm = np.sum(w, axis=1)

        def fsa(X):
            return np.sum(X * w, axis=1) / norm

        self.dl = dl
        self.Bpol = Bpol
        self.B = B

        self.B2_avg = fsa(B**2)
        self.B_avg = fsa(B)
        self.Bpol2_avg = fsa(Bpol**2)
        self.R2inv_avg = fsa(1.0 / R**2)
        self.R_avg = fsa(R)
        self.Bmax = np.max(B, axis=1)
        self.Bmin = np.min(B, axis=1)

        # Safety factor: q = (1/2pi) int f dl / (R^2 |Bpol|)
        self.q = self.fpol * norm * self.R2inv_avg / (2.0 * np.pi)

        # dV/dpsi = 2 pi int dl / |Bpol|
        #
        # The volume element is dV = 2 pi R dl dpsi / |grad psi|, and
        # |grad psi| = R |Bpol|, so the R factors cancel.
        self.dVdpsi = 2.0 * np.pi * norm

        self.Rmax = np.max(R, axis=1)
        self.Rmin = np.min(R, axis=1)
        self.Rgeo = 0.5 * (self.Rmax + self.Rmin)
        self.eps = (self.Rmax - self.Rmin) / (self.Rmax + self.Rmin)

        # Trapped fraction
        G_over_B2 = fsa(_G(B / self.Bmax[:, None]) / B**2)
        self.ft = trapped_fraction(self.B2_avg, self.B_avg, self.Bmax, G_over_B2)

    # ------------------------------------------------------------------

    def average(self, X):
        """
        Flux surface average of an (npsi, ntheta) array evaluated on the
        traced surface points.
        """
        X = np.asarray(X)
        if X.shape != self.R.shape:
            raise ValueError(f"Expected array of shape {self.R.shape}, got {X.shape}")
        Bpol_safe = np.maximum(self.Bpol, 1.0e-10 * np.max(self.B))
        w = self.dl / Bpol_safe
        return np.sum(X * w, axis=1) / np.sum(w, axis=1)

    def check(self, verbose=False):
        """
        Sanity check the traced surfaces.

        Returns a list of warning strings; also issues them as warnings if
        ``verbose`` is True. Catches the failure modes that matter for the
        neoclassical calculation: surfaces that failed to close, a
        non-monotonic or vanishing inverse aspect ratio, and trapped
        fractions outside the physical range.
        """
        messages = []

        # A traced point that collapsed onto the axis means the ray search
        # failed for that angle
        dist = np.hypot(self.R - self.Raxis, self.Z - self.Zaxis)
        collapsed = np.any(dist < 1.0e-8, axis=1)
        if np.any(collapsed):
            bad = self.psinorm[collapsed]
            messages.append(
                f"{collapsed.sum()} surface(s) have points collapsed onto the "
                f"magnetic axis, first at psi_n = {bad[0]:.4f}"
            )

        if np.any(self.eps <= 0.0):
            messages.append("Non-positive inverse aspect ratio on some surfaces")

        if np.any(np.diff(self.eps) < 0.0):
            n = int(np.sum(np.diff(self.eps) < 0.0))
            messages.append(
                f"Inverse aspect ratio is non-monotonic ({n} decreasing "
                "interval(s)); flux surface tracing may be inaccurate"
            )

        if np.any(self.ft <= 0.0) or np.any(self.ft >= 1.0):
            messages.append("Trapped fraction hit the [0, 1] clip range")

        if np.any(~np.isfinite(self.q)):
            messages.append("Non-finite safety factor on some surfaces")

        if verbose:
            for m in messages:
                warnings.warn(m, stacklevel=2)

        return messages
