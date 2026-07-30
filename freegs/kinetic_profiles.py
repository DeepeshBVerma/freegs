"""
Temperature and density profiles of the plasma species.

Provides KineticProfiles, which turns tabulated electron density, electron
temperature, ion temperature and effective charge into smooth, positive,
differentiable functions of normalised poloidal flux, together with the
derived quantities needed by the Sauter neoclassical formulas (see
freegs.sauter): main ion and impurity densities, electron and total
pressures, and logarithmic gradients.

Densities and temperatures are represented as cubic splines of their
logarithm. That has three properties which matter here:

1. The profiles are positive everywhere by construction, so no amount of
   spline ringing can produce a negative density or temperature.
2. The logarithmic gradients dln(n)/dpsi and dln(T)/dpsi that the bootstrap
   formulas need are the spline derivatives directly, rather than a ratio of
   two numerically differentiated quantities. They are therefore as smooth as
   the spline itself.
3. Extrapolation beyond the supplied data is log-linear, which is continuous
   in value and first derivative and cannot change sign.

The class also validates the profiles, with particular attention to the edge
region psi_n -> 1 where the neoclassical formulas are most fragile: the
Coulomb logarithms in Eqs. (18d,e) of Sauter et al. diverge as T -> 0, and
the collisionalities in Eqs. (18b,c) scale as n/T^2, so a profile which
collapses or rings at the boundary produces nonsense there. Use
``report()`` or ``plot()`` to inspect the outcome.

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

from .sauter import ELEMENTARY_CHARGE


class ProfileValidationError(ValueError):
    """Raised when input profiles cannot be used at all."""


class ProfileWarning(UserWarning):
    """Issued when input profiles are usable but questionable."""


def _fit_spline(x, y, smooth):
    """
    Fit a cubic spline to (x, y).

    smooth:
        None    - interpolating spline
        "auto"  - generalised cross-validation smoothing spline
        float   - UnivariateSpline smoothing factor s

    Returns an object with __call__(x) and derivative().
    """
    if smooth is None:
        return interpolate.CubicSpline(x, y, extrapolate=True)
    if isinstance(smooth, str):
        if smooth != "auto":
            raise ValueError(f"Unknown smooth option {smooth!r}")
        return interpolate.make_smoothing_spline(x, y)
    return interpolate.UnivariateSpline(x, y, s=float(smooth), k=3)


class _LogSpline:
    """
    A strictly positive profile, represented as a spline of log(value).

    Outside the range of the input data the profile is continued log-linearly
    using the end value and end derivative, which keeps it positive and C1.
    """

    def __init__(self, x, y, smooth=None, name=""):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if np.any(y <= 0.0):
            bad = x[y <= 0.0]
            raise ProfileValidationError(
                f"{name}: {len(bad)} non-positive value(s), first at "
                f"psi_n = {bad[0]:.4f}. Densities and temperatures must be "
                "strictly positive; clip or replace the edge data."
            )

        self.name = name
        self.x = x
        self.y = y
        self.xmin, self.xmax = x[0], x[-1]

        self._log = _fit_spline(x, np.log(y), smooth)
        self._dlog = self._log.derivative()

        # End states for log-linear extrapolation
        self._log_lo = float(self._log(self.xmin))
        self._dlog_lo = float(self._dlog(self.xmin))
        self._log_hi = float(self._log(self.xmax))
        self._dlog_hi = float(self._dlog(self.xmax))

    def logvalue(self, xq):
        xq = np.asarray(xq, dtype=float)
        inner = np.clip(xq, self.xmin, self.xmax)
        out = np.asarray(self._log(inner), dtype=float)
        out = np.where(
            xq < self.xmin, self._log_lo + self._dlog_lo * (xq - self.xmin), out
        )
        return np.where(
            xq > self.xmax, self._log_hi + self._dlog_hi * (xq - self.xmax), out
        )

    def __call__(self, xq):
        return np.exp(self.logvalue(xq))

    def dlog(self, xq):
        """dln(value)/dx"""
        xq = np.asarray(xq, dtype=float)
        inner = np.clip(xq, self.xmin, self.xmax)
        out = np.asarray(self._dlog(inner), dtype=float)
        out = np.where(xq < self.xmin, self._dlog_lo, out)
        return np.where(xq > self.xmax, self._dlog_hi, out)

    def derivative(self, xq):
        """d(value)/dx"""
        return self(xq) * self.dlog(xq)


class _Spline:
    """A plain spline with clamped-value, C1 linear extrapolation."""

    def __init__(self, x, y, smooth=None, name=""):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.name = name
        self.x = x
        self.y = y
        self.xmin, self.xmax = x[0], x[-1]
        self._s = _fit_spline(x, y, smooth)
        self._ds = self._s.derivative()
        self._lo = float(self._s(self.xmin))
        self._dlo = float(self._ds(self.xmin))
        self._hi = float(self._s(self.xmax))
        self._dhi = float(self._ds(self.xmax))

    def __call__(self, xq):
        xq = np.asarray(xq, dtype=float)
        inner = np.clip(xq, self.xmin, self.xmax)
        out = np.asarray(self._s(inner), dtype=float)
        out = np.where(xq < self.xmin, self._lo + self._dlo * (xq - self.xmin), out)
        return np.where(xq > self.xmax, self._hi + self._dhi * (xq - self.xmax), out)

    def derivative(self, xq):
        xq = np.asarray(xq, dtype=float)
        inner = np.clip(xq, self.xmin, self.xmax)
        out = np.asarray(self._ds(inner), dtype=float)
        out = np.where(xq < self.xmin, self._dlo, out)
        return np.where(xq > self.xmax, self._dhi, out)


class _Constant:
    """A constant profile, matching the _Spline interface."""

    def __init__(self, value, name=""):
        self.name = name
        self.value = float(value)
        self.x = np.array([0.0, 1.0])
        self.y = np.array([self.value, self.value])
        self.xmin, self.xmax = 0.0, 1.0

    def __call__(self, xq):
        return np.full(np.shape(xq), self.value)

    def derivative(self, xq):
        return np.zeros(np.shape(xq))


class KineticProfiles:
    """
    Density and temperature profiles of the plasma species.

    Species model: electrons, one main ion of charge ``ion_Z``, and one
    impurity of charge ``impurity_Z``. The main ion and impurity densities
    follow from quasineutrality and the definition of Z_eff,

        n_e            = Z_i n_i + Z_imp n_imp
        n_e Z_eff      = Z_i^2 n_i + Z_imp^2 n_imp

    so that

        n_i   = n_e (Z_imp - Z_eff) / ( Z_i (Z_imp - Z_i) )
        n_imp = n_e (Z_eff - Z_i)   / ( Z_imp (Z_imp - Z_i) )

    Both ion species are assumed to be at the ion temperature T_i.

    Units: densities in m^-3, temperatures in eV, pressures in Pa. Radial
    coordinate is normalised poloidal flux psi_n, 0 on the magnetic axis and
    1 on the plasma boundary.
    """

    def __init__(
        self,
        psi_n,
        ne,
        Te,
        Ti=None,
        Zeff=1.0,
        ion_Z=1.0,
        ion_A=2.0,
        impurity_Z=6.0,
        smooth=None,
        validate=True,
        ne_floor=1.0e16,
        Te_floor=1.0,
        edge_start=0.9,
    ):
        """
        Parameters
        ----------
        psi_n:
            1D array of normalised poloidal flux, strictly increasing. Should
            reach 1.0; if it stops short, the edge is extrapolated and a
            warning is issued.
        ne:
            Electron density [m^-3]
        Te:
            Electron temperature [eV]
        Ti:
            Ion temperature [eV]. Defaults to Te if not given.
        Zeff:
            Effective charge. Scalar, or an array matching psi_n. Must lie
            between ion_Z and impurity_Z for the densities to be positive.
        ion_Z, ion_A:
            Charge and mass number of the main ion (default deuterium).
        impurity_Z:
            Charge of the impurity species used to carry Z_eff
            (default 6, carbon).
        smooth:
            Spline smoothing. None for an interpolating spline, a float for a
            UnivariateSpline smoothing factor applied to the logarithm of the
            profile, or "auto" for generalised cross-validation. Use a
            nonzero value for noisy experimental data, where an interpolating
            spline will ring and produce spurious gradients.
        validate:
            Run the consistency and smoothness checks on construction.
        ne_floor, Te_floor:
            Minimum acceptable edge density [m^-3] and temperature [eV].
        edge_start:
            psi_n above which the "edge region" checks apply.
        """
        psi_n = np.asarray(psi_n, dtype=float)

        if psi_n.ndim != 1 or psi_n.size < 4:
            raise ProfileValidationError(
                "psi_n must be a 1D array with at least 4 points"
            )
        if not np.all(np.isfinite(psi_n)):
            raise ProfileValidationError("psi_n contains non-finite values")
        if np.any(np.diff(psi_n) <= 0.0):
            raise ProfileValidationError("psi_n must be strictly increasing")

        ne = np.asarray(ne, dtype=float)
        Te = np.asarray(Te, dtype=float)
        Ti = Te.copy() if Ti is None else np.asarray(Ti, dtype=float)

        for name, arr in (("ne", ne), ("Te", Te), ("Ti", Ti)):
            if arr.shape != psi_n.shape:
                raise ProfileValidationError(
                    f"{name} has shape {arr.shape}, expected {psi_n.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ProfileValidationError(f"{name} contains non-finite values")

        if impurity_Z <= ion_Z:
            raise ProfileValidationError(
                f"impurity_Z ({impurity_Z}) must exceed ion_Z ({ion_Z})"
            )

        self.psi_n_data = psi_n
        self.ion_Z = float(ion_Z)
        self.ion_A = float(ion_A)
        self.impurity_Z = float(impurity_Z)
        self.ne_floor = float(ne_floor)
        self.Te_floor = float(Te_floor)
        self.edge_start = float(edge_start)
        self.smooth = smooth

        self._ne = _LogSpline(psi_n, ne, smooth, name="ne")
        self._Te = _LogSpline(psi_n, Te, smooth, name="Te")
        self._Ti = _LogSpline(psi_n, Ti, smooth, name="Ti")

        Zeff_arr = np.asarray(Zeff, dtype=float)
        if Zeff_arr.ndim == 0:
            self._Zeff = _Constant(float(Zeff_arr), name="Zeff")
            self.Zeff_data = np.full_like(psi_n, float(Zeff_arr))
        else:
            if Zeff_arr.shape != psi_n.shape:
                raise ProfileValidationError(
                    f"Zeff has shape {Zeff_arr.shape}, expected {psi_n.shape}"
                )
            self._Zeff = _Spline(psi_n, Zeff_arr, smooth, name="Zeff")
            self.Zeff_data = Zeff_arr

        # dilution coefficient c(Zeff) with n_ion_total = ne * c, and its
        # derivative with respect to Zeff, which is a constant
        self._dc_dZeff = -1.0 / (self.ion_Z * self.impurity_Z)

        self.messages = []
        if validate:
            self.messages = self.validate()

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------

    def ne(self, psi_n):
        """Electron density [m^-3]"""
        return self._ne(psi_n)

    def Te(self, psi_n):
        """Electron temperature [eV]"""
        return self._Te(psi_n)

    def Ti(self, psi_n):
        """Ion temperature [eV]"""
        return self._Ti(psi_n)

    def Zeff(self, psi_n):
        """Effective charge"""
        return np.clip(self._Zeff(psi_n), self.ion_Z, self.impurity_Z)

    def _dilution(self, psi_n):
        """
        Return (c, dc/dpsi_n) where n_i + n_imp = ne * c.

        c = (Z_imp - Zeff)/(Z_i (Z_imp - Z_i)) + (Zeff - Z_i)/(Z_imp (Z_imp - Z_i))
        """
        Zi, Zimp = self.ion_Z, self.impurity_Z
        Zeff = self.Zeff(psi_n)
        c = (Zimp - Zeff) / (Zi * (Zimp - Zi)) + (Zeff - Zi) / (Zimp * (Zimp - Zi))

        # Zero the derivative where the clip is active, so c and dc agree
        raw = self._Zeff(psi_n)
        clipped = (raw < Zi) | (raw > Zimp)
        dZeff = np.where(clipped, 0.0, self._Zeff.derivative(psi_n))
        return c, self._dc_dZeff * dZeff

    def ni(self, psi_n):
        """Main ion density [m^-3]"""
        Zi, Zimp = self.ion_Z, self.impurity_Z
        return self.ne(psi_n) * (Zimp - self.Zeff(psi_n)) / (Zi * (Zimp - Zi))

    def nimp(self, psi_n):
        """Impurity density [m^-3]"""
        Zi, Zimp = self.ion_Z, self.impurity_Z
        return self.ne(psi_n) * (self.Zeff(psi_n) - Zi) / (Zimp * (Zimp - Zi))

    def n_ion(self, psi_n):
        """Total ion density n_i + n_imp [m^-3]"""
        c, _ = self._dilution(psi_n)
        return self.ne(psi_n) * c

    # ------------------------------------------------------------------
    # Pressures
    # ------------------------------------------------------------------

    def pe(self, psi_n):
        """Electron pressure [Pa]"""
        return ELEMENTARY_CHARGE * self.ne(psi_n) * self.Te(psi_n)

    def pi(self, psi_n):
        """Total ion pressure [Pa]"""
        return ELEMENTARY_CHARGE * self.n_ion(psi_n) * self.Ti(psi_n)

    def pressure(self, psi_n):
        """Total pressure p = p_e + p_i [Pa]"""
        c, _ = self._dilution(psi_n)
        return (
            ELEMENTARY_CHARGE * self.ne(psi_n) * (self.Te(psi_n) + c * self.Ti(psi_n))
        )

    def dpressure_dpsin(self, psi_n):
        """
        dp/dpsi_n [Pa], evaluated analytically from the component splines.

        p = e n_e (T_e + c T_i), so

        dp/dx = e [ n_e' (T_e + c T_i) + n_e (T_e' + c' T_i + c T_i') ]
        """
        ne = self.ne(psi_n)
        Te = self.Te(psi_n)
        Ti = self.Ti(psi_n)
        c, dc = self._dilution(psi_n)

        dne = ne * self._ne.dlog(psi_n)
        dTe = Te * self._Te.dlog(psi_n)
        dTi = Ti * self._Ti.dlog(psi_n)

        return ELEMENTARY_CHARGE * (
            dne * (Te + c * Ti) + ne * (dTe + dc * Ti + c * dTi)
        )

    def Rpe(self, psi_n):
        """Ratio p_e / p, called R_pe in Sauter et al. Eq. (6)."""
        return self.pe(psi_n) / self.pressure(psi_n)

    # ------------------------------------------------------------------
    # Logarithmic gradients with respect to psi_n
    # ------------------------------------------------------------------

    def dlnne_dpsin(self, psi_n):
        """dln(n_e)/dpsi_n"""
        return self._ne.dlog(psi_n)

    def dlnTe_dpsin(self, psi_n):
        """dln(T_e)/dpsi_n"""
        return self._Te.dlog(psi_n)

    def dlnTi_dpsin(self, psi_n):
        """dln(T_i)/dpsi_n"""
        return self._Ti.dlog(psi_n)

    def dlnp_dpsin(self, psi_n):
        """dln(p)/dpsi_n for the total pressure"""
        return self.dpressure_dpsin(psi_n) / self.pressure(psi_n)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, ndense=401, warn=True):
        """
        Check the profiles for problems, with emphasis on the edge.

        Raises ProfileValidationError for conditions which make the
        neoclassical calculation impossible, and returns a list of warning
        strings for conditions which merely make it suspect. If ``warn`` is
        True the warnings are also issued as ProfileWarning.

        Checks performed:

        - Z_eff lies between ion_Z and impurity_Z, so that both ion
          densities are positive
        - the data reaches psi_n = 1, otherwise the edge is extrapolated
        - edge density and temperature exceed the supplied floors
        - the fitted splines do not overshoot the range of the input data,
          which would indicate ringing
        - logarithmic gradients are finite and bounded
        - the edge region contains enough data points to resolve the
          gradient scale length there
        - the total pressure is positive and its gradient does not change
          sign spuriously at the boundary
        """
        messages = []
        x = self.psi_n_data

        # --- Z_eff consistency: negative ion densities are fatal
        Zraw = np.asarray(self._Zeff(x), dtype=float)
        if np.any(Zraw < self.ion_Z - 1e-9) or np.any(Zraw > self.impurity_Z + 1e-9):
            lo, hi = float(np.min(Zraw)), float(np.max(Zraw))
            raise ProfileValidationError(
                f"Zeff ranges over [{lo:.3f}, {hi:.3f}], outside "
                f"[ion_Z, impurity_Z] = [{self.ion_Z}, {self.impurity_Z}]. "
                "This gives a negative main ion or impurity density. Either "
                "clip Zeff or choose a different impurity_Z."
            )

        # --- Does the data reach the boundary?
        if x[-1] < 1.0 - 1e-6:
            messages.append(
                f"Profile data stops at psi_n = {x[-1]:.4f}, so the region "
                f"{x[-1]:.4f} < psi_n <= 1 is log-linearly extrapolated. The "
                "bootstrap current there is an extrapolation, not data."
            )
        if x[0] > 0.05:
            messages.append(
                f"Profile data starts at psi_n = {x[0]:.4f}; the core inside "
                "that is extrapolated."
            )

        # --- Edge magnitudes
        ne1 = float(self.ne(1.0))
        Te1 = float(self.Te(1.0))
        Ti1 = float(self.Ti(1.0))
        if ne1 < self.ne_floor:
            messages.append(
                f"Edge density n_e(1) = {ne1:.3e} m^-3 is below the floor "
                f"{self.ne_floor:.3e}. The Coulomb logarithm and "
                "collisionality at the boundary are unreliable."
            )
        if Te1 < self.Te_floor:
            messages.append(
                f"Edge electron temperature T_e(1) = {Te1:.3f} eV is below "
                f"the floor {self.Te_floor:.3f}. nu*_e ~ n/T^2 will be huge "
                "and the conductivity ~T^(3/2) near zero."
            )
        if Ti1 < self.Te_floor:
            messages.append(
                f"Edge ion temperature T_i(1) = {Ti1:.3f} eV is below the "
                f"floor {self.Te_floor:.3f}."
            )

        # --- Spline fidelity on a dense grid
        xd = np.linspace(x[0], min(x[-1], 1.0), ndense)
        for name, spl, data in (
            ("ne", self._ne, self._ne.y),
            ("Te", self._Te, self._Te.y),
            ("Ti", self._Ti, self._Ti.y),
        ):
            dense = spl(xd)
            if not np.all(np.isfinite(dense)):
                raise ProfileValidationError(
                    f"{name}: fitted spline is not finite over [0, 1]"
                )

            dlo, dhi = float(np.min(data)), float(np.max(data))
            span = dhi - dlo
            if span > 0.0:
                over = (float(np.max(dense)) - dhi) / span
                under = (dlo - float(np.min(dense))) / span
                if over > 0.05 or under > 0.05:
                    messages.append(
                        f"{name}: fitted spline overshoots the input data by "
                        f"{100 * max(over, under):.1f}% of its range. This is "
                        "spline ringing; pass smooth= to damp it or supply a "
                        "smoother profile."
                    )

            dlog = spl.dlog(xd)
            if not np.all(np.isfinite(dlog)):
                raise ProfileValidationError(
                    f"{name}: dln/dpsi_n is not finite over [0, 1]"
                )
            gmax = float(np.max(np.abs(dlog)))
            if gmax > 1.0e3:
                messages.append(
                    f"{name}: |dln({name})/dpsi_n| reaches {gmax:.3e}, which "
                    "is effectively a discontinuity. The bootstrap current is "
                    "proportional to this gradient and will spike."
                )

        # --- Edge region resolution and behaviour
        edge = x >= self.edge_start
        n_edge = int(np.count_nonzero(edge))
        if n_edge < 4:
            messages.append(
                f"Only {n_edge} data point(s) at psi_n >= {self.edge_start}. "
                "A pedestal cannot be resolved; the edge gradients driving "
                "the bootstrap current are set by the spline, not the data."
            )
        else:
            dx_edge = float(np.max(np.diff(x[edge])))
            xe = np.linspace(self.edge_start, min(x[-1], 1.0), 101)
            worst = 0.0
            worst_name = ""
            for name, spl in (("ne", self._ne), ("Te", self._Te), ("Ti", self._Ti)):
                g = float(np.max(np.abs(spl.dlog(xe))))
                if g > worst:
                    worst, worst_name = g, name
            if worst > 0.0:
                scale_length = 1.0 / worst
                if dx_edge > 0.5 * scale_length:
                    messages.append(
                        f"Edge grid spacing {dx_edge:.4f} in psi_n exceeds half "
                        f"the shortest gradient scale length "
                        f"({scale_length:.4f}, set by {worst_name}). The edge "
                        "gradient is under-resolved."
                    )

        # --- Turning points introduced at the very edge are a ringing signature
        xtail = np.linspace(max(self.edge_start, x[0]), min(x[-1], 1.0), 201)
        for name, spl in (("ne", self._ne), ("Te", self._Te), ("Ti", self._Ti)):
            g = spl.dlog(xtail)
            sign_changes = int(np.count_nonzero(np.diff(np.sign(g)) != 0))
            # Compare against the data's own turning points in the same window
            sel = (x >= xtail[0]) & (x <= xtail[-1])
            if np.count_nonzero(sel) >= 3:
                gdata = np.diff(np.log(spl.y[sel]))
                data_changes = int(np.count_nonzero(np.diff(np.sign(gdata)) != 0))
                if sign_changes > data_changes + 1:
                    messages.append(
                        f"{name}: fitted gradient changes sign {sign_changes} "
                        f"time(s) for psi_n > {xtail[0]:.2f}, against "
                        f"{data_changes} in the data. Likely spline ringing "
                        "at the edge."
                    )

        # --- Pressure sanity
        p = self.pressure(xd)
        if np.any(p <= 0.0):
            raise ProfileValidationError("Total pressure is not positive")
        dp = self.dpressure_dpsin(xd)
        if np.any(dp[xd > self.edge_start] > 0.0) and np.all(dp[xd < 0.5] < 0.0):
            messages.append(
                "Pressure gradient changes sign in the edge region while the "
                "core is monotonically decreasing. Check the edge data: this "
                "reverses the sign of the local bootstrap current."
            )

        if warn:
            for m in messages:
                warnings.warn(m, ProfileWarning, stacklevel=2)

        return messages

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, psi_n=(0.0, 0.5, 0.9, 0.95, 0.99, 1.0)):
        """Return a formatted table of the profiles at selected psi_n."""
        psi_n = np.asarray(psi_n, dtype=float)
        header = (
            "  psi_n        ne [m^-3]     Te [eV]     Ti [eV]     Zeff"
            "        ni [m^-3]      p [Pa]   dlnp/dpsin"
        )
        lines = [header]
        for x in psi_n:
            lines.append(
                f"  {x:6.3f}   {float(self.ne(x)):12.4e} {float(self.Te(x)):11.2f} "
                f"{float(self.Ti(x)):11.2f} {float(self.Zeff(x)):8.3f} "
                f"{float(self.ni(x)):14.4e} {float(self.pressure(x)):11.2f} "
                f"{float(self.dlnp_dpsin(x)):12.4f}"
            )
        return "\n".join(lines)

    def report(self, psi_n=(0.0, 0.5, 0.9, 0.95, 0.99, 1.0)):
        """Print the profile summary and any validation messages."""
        print("Kinetic profiles")
        print(
            f"  main ion Z = {self.ion_Z:g}, A = {self.ion_A:g};  "
            f"impurity Z = {self.impurity_Z:g}"
        )
        print(
            f"  data on psi_n in [{self.psi_n_data[0]:.4f}, "
            f"{self.psi_n_data[-1]:.4f}] with {len(self.psi_n_data)} points; "
            f"smooth={self.smooth!r}"
        )
        print()
        print(self.summary(psi_n))
        print()
        if self.messages:
            print(f"{len(self.messages)} validation warning(s):")
            for i, m in enumerate(self.messages, 1):
                print(f"  {i}. {m}")
        else:
            print("Validation: no warnings.")

    def plot(self, axes=None, show=True, npoints=401):
        """
        Plot the profiles, their logarithmic gradients, and an edge zoom.

        Input data points are overlaid so that spline ringing and edge
        extrapolation are visible by eye.
        """
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = np.asarray(axes).reshape(2, 3)

        x = np.linspace(0.0, 1.0, npoints)
        xd = self.psi_n_data

        panels = (
            ("$n_e$ [m$^{-3}$]", self.ne, self._ne.y),
            ("$T_e$, $T_i$ [eV]", None, None),
            ("$p$ [Pa]", self.pressure, None),
        )

        ax = axes[0, 0]
        ax.plot(x, self.ne(x), "-")
        ax.plot(xd, self._ne.y, "k.", ms=4)
        ax.set_ylabel(panels[0][0])

        ax = axes[0, 1]
        ax.plot(x, self.Te(x), "-", label="$T_e$")
        ax.plot(x, self.Ti(x), "-", label="$T_i$")
        ax.plot(xd, self._Te.y, "k.", ms=4)
        ax.plot(xd, self._Ti.y, "k.", ms=4)
        ax.set_ylabel("$T$ [eV]")
        ax.legend()

        ax = axes[0, 2]
        ax.plot(x, self.pressure(x), "-", label="$p$")
        ax.plot(x, self.pe(x), "--", label="$p_e$")
        ax.set_ylabel("pressure [Pa]")
        ax.legend()

        ax = axes[1, 0]
        ax.plot(x, self.dlnne_dpsin(x), label=r"$d\ln n_e/d\psi_N$")
        ax.plot(x, self.dlnTe_dpsin(x), label=r"$d\ln T_e/d\psi_N$")
        ax.plot(x, self.dlnTi_dpsin(x), label=r"$d\ln T_i/d\psi_N$")
        ax.plot(x, self.dlnp_dpsin(x), "k", label=r"$d\ln p/d\psi_N$")
        ax.set_ylabel("logarithmic gradient")
        ax.legend(fontsize="small")

        # Edge zoom, log scale, to expose collapse or ringing near psi_n = 1
        xe = np.linspace(self.edge_start, 1.0, npoints)
        ax = axes[1, 1]
        ax.semilogy(xe, self.ne(xe), label="$n_e$")
        sel = xd >= self.edge_start
        ax.semilogy(xd[sel], self._ne.y[sel], "k.", ms=5)
        ax.axhline(self.ne_floor, color="r", ls=":", label="floor")
        ax.set_ylabel("$n_e$ [m$^{-3}$] (edge)")
        ax.legend(fontsize="small")

        ax = axes[1, 2]
        ax.semilogy(xe, self.Te(xe), label="$T_e$")
        ax.semilogy(xe, self.Ti(xe), label="$T_i$")
        ax.semilogy(xd[sel], self._Te.y[sel], "k.", ms=5)
        ax.axhline(self.Te_floor, color="r", ls=":", label="floor")
        ax.set_ylabel("$T$ [eV] (edge)")
        ax.legend(fontsize="small")

        for ax in axes.flat:
            ax.set_xlabel(r"$\psi_N$")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        return axes


def mtanh_profile(
    psi_n,
    core,
    ped,
    sep,
    ped_position=0.94,
    ped_width=0.05,
    alpha=1.5,
    beta=2.0,
):
    """
    A smooth core-plus-pedestal profile, useful for tests and scans.

    y(x) = sep + (ped - sep) * 0.5 * (1 - tanh((x - x_ped) / (w/2)))
           + (core - ped) * max(0, 1 - (x/x_ped)^alpha)^beta

    The hyperbolic tangent term gives a pedestal of width ``ped_width``
    centred on ``ped_position``, decaying to ``sep`` at the boundary; the
    second term adds a core profile which vanishes at the pedestal top.

    By construction this is smooth, strictly positive for positive ``sep``,
    and has bounded gradients at psi_n = 1, so it will not trip the edge
    checks in KineticProfiles.

    Parameters
    ----------
    psi_n:
        Normalised poloidal flux
    core:
        On-axis value
    ped:
        Pedestal top value
    sep:
        Separatrix value. Must be positive.
    ped_position, ped_width:
        Centre and width of the pedestal in psi_n
    alpha, beta:
        Core profile shape exponents
    """
    if sep <= 0.0:
        raise ValueError("mtanh_profile needs a positive separatrix value")

    x = np.asarray(psi_n, dtype=float)
    edge = sep + (ped - sep) * 0.5 * (
        1.0 - np.tanh((x - ped_position) / (0.5 * ped_width))
    )
    shape = (
        np.clip(1.0 - np.clip(x / ped_position, 0.0, None) ** alpha, 0.0, None) ** beta
    )
    return edge + (core - ped) * shape
