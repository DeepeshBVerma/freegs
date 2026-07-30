"""
Bootstrap-consistent plasma profiles for FreeGS.

Provides BootstrapProfiles, a freegs.jtor.Profile implementation in which the
current profile is not prescribed by a shape function but derived from the
neoclassical parallel current of

    O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6, 2834 (1999)

given electron and ion density and temperature profiles (see
freegs.kinetic_profiles).

How it works
------------

The pressure is fixed outright by the kinetic profiles, p = p_e + p_i, so
p'(psi) is known and never fitted. What remains is ff'(psi), and that is
obtained from the flux surface averaged parallel current. Writing
B = f grad(phi) + grad(psi) x grad(phi) and using the Grad-Shafranov
equation, mu0 J_pol = f' B_pol and J_phi = R p' + ff'/(mu0 R), so

    J.B = f p' + (f'/mu0) B^2

holds pointwise. Flux surface averaging and rearranging,

    ff' = mu0 f ( <j.B> - f p' ) / <B^2>                             (*)

Sauter's Eq. (5) supplies <j.B> as the sum of an inductive and a bootstrap
term, so ff' splits into three exactly additive pieces:

    ff'_ohm = mu0 f <j.B>_ohm / <B^2>       with <j.B>_ohm = sigma_neo <E.B>
    ff'_bs  = mu0 f <j.B>_bs  / <B^2>
    ff'_dia = -mu0 f^2 p' / <B^2>

and correspondingly the toroidal current density splits as

    Jtor_ohm = ff'_ohm / (mu0 R)
    Jtor_bs  = ff'_bs  / (mu0 R)
    Jtor_dia = R p' + ff'_dia / (mu0 R)

whose sum is the total Jtor. Integrating each piece over the plasma
cross-section gives I_ohm + I_bs + I_dia = Ip exactly, which is the
decomposition reported by BootstrapAnalysis.

The inductive field is written in terms of a loop voltage,
<E.B> = Vloop f <1/R^2> / (2 pi). Since <j.B>_ohm is linear in Vloop and (*)
is linear in <j.B>, the total plasma current is an affine function of Vloop,
so the Vloop which meets the target Ip is found in closed form at every
Picard iteration rather than by iteration.

Edge handling
-------------

Sauter's collisionalities, Eqs. (18b,c), contain q R / eps^(3/2). Both q and
the flux surface tracing degrade as psi_n -> 1, where q diverges at the
X-point. Left alone this drives nu* -> infinity, hence f_t,eff -> 0 and a
spurious collapse of the bootstrap current exactly where the pedestal
gradient is largest. Here the geometric and collisionality inputs
(q, eps, f_t, <B^2>, <1/R^2>, f, R) are frozen at their values on the last
reliably traced surface, ``psi_max``, while the local kinetic gradients
continue to be evaluated from the profiles. The frozen values are reported
by ``BootstrapAnalysis.report()`` so the effect of the cutoff is visible.

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
from scipy.integrate import romb, trapezoid

from . import critical, sauter
from .fluxsurface import FluxSurfaces
from .gradshafranov import mu0
from .jtor import Profile


def make_psin_grid(npsi, psi_min, psi_max):
    """
    Build the 1D normalised flux grid used for neoclassical calculations.

    The grid is the magnetic axis, then ``npsi`` surfaces from ``psi_min`` to
    ``psi_max`` which are actually traced, then a few more out to psi_n = 1
    which reuse the geometry frozen at ``psi_max``.

    Returns
    -------
    (psi_n, traced_slice):
        The grid, and the slice of it corresponding to traced surfaces.
    """
    if npsi < 8:
        raise ValueError("npsi must be at least 8")
    if not 0.0 < psi_min < psi_max < 1.0:
        raise ValueError("Require 0 < psi_min < psi_max < 1")

    traced = np.linspace(psi_min, psi_max, npsi)
    n_edge = max(4, round(npsi * (1.0 - psi_max) / (psi_max - psi_min)) + 4)
    edge = np.linspace(psi_max, 1.0, n_edge + 1)[1:]
    return np.concatenate(([0.0], traced, edge)), slice(1, 1 + npsi)


def neoclassical_quantities(
    eq,
    kinetic,
    psi_n,
    traced_slice,
    fpol,
    psi=None,
    psi_axis=None,
    psi_bndry=None,
    ntheta=128,
    nray=256,
    opoint=None,
    xpoint=None,
    check_surfaces=True,
):
    """
    Flux surface geometry, Sauter coefficients and the bootstrap <j.B>.

    This is the shared core used both by BootstrapProfiles, where it is
    evaluated at every Picard iteration, and by analyse_equilibrium, where it
    is evaluated once on a fixed equilibrium.

    Parameters
    ----------
    eq:
        Equilibrium supplying the geometry
    kinetic:
        KineticProfiles instance
    psi_n, traced_slice:
        As returned by make_psin_grid
    fpol:
        f = R*Btor on psi_n. Passed in rather than taken from eq because
        during a Picard iteration the relevant f is the current iterate's,
        which is not yet stored on the Equilibrium.
    psi, psi_axis, psi_bndry:
        Flux array and its axis/boundary values. Default to the equilibrium's.
    ntheta, nray, opoint, xpoint:
        Passed to FluxSurfaces
    check_surfaces:
        Warn if the traced surfaces look unreliable

    Returns
    -------
    dict of 1D arrays, all of length len(psi_n), plus the key "surface_messages".
    """
    x = np.asarray(psi_n, dtype=float)
    fpol = np.asarray(fpol, dtype=float)

    if psi is None:
        psi = eq.psi()
    if psi_axis is None:
        psi_axis = eq.psi_axis
    if psi_bndry is None:
        psi_bndry = eq.psi_bndry
    dpsi = psi_bndry - psi_axis

    n_traced = traced_slice.stop - traced_slice.start

    # --- Flux surface geometry on the traced surfaces
    fs = FluxSurfaces(
        eq,
        x[traced_slice],
        psi=psi,
        ntheta=ntheta,
        nray=nray,
        psi_axis=psi_axis,
        psi_bndry=psi_bndry,
        opoint=opoint,
        xpoint=xpoint,
    )
    messages = []
    if check_surfaces:
        messages = fs.check()
        for m in messages:
            warnings.warn(f"Flux surface tracing: {m}", stacklevel=3)

    # --- Extend onto the full grid.
    #
    # Index 0 is the axis: geometry frozen at the innermost traced surface,
    # trapped fraction forced to zero, which makes the bootstrap current
    # vanish there and sigma_neo reduce to Spitzer, both correct on axis.
    # Indices beyond the traced range: everything frozen at psi_max.
    def extend(values, axis_value=None):
        out = np.empty(len(x))
        out[traced_slice] = values
        out[0] = values[0] if axis_value is None else axis_value
        out[traced_slice.stop :] = values[-1]
        return out

    ft = extend(fs.ft, axis_value=0.0)
    eps = extend(fs.eps)
    q = extend(fs.q)
    Rgeo = extend(fs.Rgeo)
    B2_avg = extend(fs.B2_avg)
    R2inv_avg = extend(fs.R2inv_avg)
    dVdpsi = extend(fs.dVdpsi)

    # --- Kinetic quantities
    ne = kinetic.ne(x)
    Te = kinetic.Te(x)
    Ti = kinetic.Ti(x)
    ni = kinetic.ni(x)
    Zeff = kinetic.Zeff(x)
    p = kinetic.pressure(x)
    pe = kinetic.pe(x)

    # Convert psi_n derivatives to psi derivatives
    dlnp_dpsi = kinetic.dlnp_dpsin(x) / dpsi
    dlnTe_dpsi = kinetic.dlnTe_dpsin(x) / dpsi
    dlnTi_dpsi = kinetic.dlnTi_dpsin(x) / dpsi
    pprime = kinetic.dpressure_dpsin(x) / dpsi

    # --- Collisionalities, Eqs. (18b,c)
    lnLe = sauter.coulomb_log_e(ne, Te)
    lnLii = sauter.coulomb_log_ii(ni, Ti, Zeff)
    nue_star = sauter.nu_star_e(ne, Te, Zeff, q, Rgeo, eps, coulomb_log=lnLe)
    nui_star = sauter.nu_star_i(ni, Ti, Zeff, q, Rgeo, eps, coulomb_log=lnLii)

    # --- Conductivity and bootstrap coefficients
    sigma = sauter.sigma_neo(ft, nue_star, Zeff, Te, coulomb_log=lnLe)
    coeffs = sauter.bootstrap_coefficients(ft, nue_star, nui_star, Zeff)

    # --- <j.B> pieces
    jdotB_bs = sauter.jdotB_bootstrap(
        fpol,
        pe,
        p,
        dlnp_dpsi,
        dlnTe_dpsi,
        dlnTi_dpsi,
        ft,
        nue_star,
        nui_star,
        Zeff,
    )
    # Ohmic term per unit loop voltage
    jdotB_ohm_unit = sauter.jdotB_ohmic(sigma, fpol, R2inv_avg, 1.0)

    return {
        "psi_n": x,
        "traced": _traced_mask(len(x), traced_slice),
        "n_traced": n_traced,
        "surface_messages": messages,
        # geometry
        "ft": ft,
        "eps": eps,
        "q": q,
        "Rgeo": Rgeo,
        "B2_avg": B2_avg,
        "R2inv_avg": R2inv_avg,
        "dVdpsi": dVdpsi,
        "fpol": fpol,
        # kinetic
        "ne": ne,
        "Te": Te,
        "Ti": Ti,
        "ni": ni,
        "Zeff": Zeff,
        "p": p,
        "pe": pe,
        "pprime": pprime,
        "dlnp_dpsi": dlnp_dpsi,
        "dlnTe_dpsi": dlnTe_dpsi,
        "dlnTi_dpsi": dlnTi_dpsi,
        # neoclassical
        "lnLambda_e": lnLe,
        "lnLambda_ii": lnLii,
        "nue_star": nue_star,
        "nui_star": nui_star,
        "sigma_neo": sigma,
        "L31": coeffs["L31"],
        "L32": coeffs["L32"],
        "L34": coeffs["L34"],
        "alpha": coeffs["alpha"],
        # currents
        "jdotB_bs": jdotB_bs,
        "jdotB_ohm_unit": jdotB_ohm_unit,
    }


def _traced_mask(n, traced_slice):
    """Boolean mask, True where the flux surface geometry was traced."""
    mask = np.zeros(n, dtype=bool)
    mask[traced_slice] = True
    return mask


class BootstrapProfiles(Profile):
    """
    Plasma profiles with a self-consistent neoclassical bootstrap current.

    The pressure comes from the supplied kinetic profiles; the current profile
    comes from the Sauter neoclassical parallel current, with the loop voltage
    set at each iteration so that the total plasma current matches ``Ip``.

    After a solve, use ``analysis()`` to obtain the ohmic / bootstrap /
    diamagnetic decomposition.
    """

    def __init__(
        self,
        eq,
        kinetic,
        Ip,
        fvac,
        npsi=64,
        ntheta=128,
        psi_min=0.02,
        psi_max=0.995,
        nray=256,
        check_surfaces=True,
    ):
        """
        Parameters
        ----------
        eq:
            The Equilibrium being solved. Needed because the neoclassical
            coefficients depend on the flux surface geometry, which is
            re-evaluated at every Picard iteration.
        kinetic:
            A KineticProfiles instance supplying n_e, T_e, T_i and Z_eff.
        Ip:
            Target total plasma current [Amps]. May be either sign.
        fvac:
            Vacuum f = R * Btor [T m]. May be either sign.
        npsi:
            Number of flux surfaces used for the 1D neoclassical calculation.
        ntheta:
            Number of poloidal points per traced flux surface.
        psi_min:
            Innermost normalised flux traced. Flux surface geometry is not
            resolvable at the magnetic axis, where eps -> 0. Inside this the
            geometry is frozen and the trapped fraction is taken to zero, so
            the bootstrap current vanishes on axis and sigma_neo reduces to
            the Spitzer value, both of which are correct there.
        psi_max:
            Outermost normalised flux traced. Above this the geometric and
            collisionality inputs are frozen (see the module docstring).
        nray:
            Ray samples used when locating flux surfaces.
        check_surfaces:
            Warn if the traced flux surfaces look unreliable.
        """
        if npsi < 8:
            raise ValueError("npsi must be at least 8")
        if not 0.0 < psi_min < psi_max < 1.0:
            raise ValueError("Require 0 < psi_min < psi_max < 1")

        self.eq = eq
        self.kinetic = kinetic
        self.Ip = float(Ip)
        self._fvac = float(fvac)
        self.npsi = int(npsi)
        self.ntheta = int(ntheta)
        self.psi_min = float(psi_min)
        self.psi_max = float(psi_max)
        self.nray = int(nray)
        self.check_surfaces = check_surfaces

        # 1D grid for the neoclassical calculation. The axis point carries
        # f_t = 0 exactly; surfaces beyond psi_max reuse frozen geometry.
        self.psi_n, self._traced_slice = make_psin_grid(npsi, psi_min, psi_max)
        self._n_traced = npsi

        # Filled in by Jtor()
        self._ffprime_spline = None
        self._ffprime_int = None
        self.psi_axis = None
        self.psi_bndry = None
        self.Vloop = None
        self.neo = {}
        self._surface_messages = []

    # ------------------------------------------------------------------
    # Grad-Shafranov right hand side
    # ------------------------------------------------------------------

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """
        Toroidal current density from the neoclassical parallel current.

        Jtor = R p' + ff'/(mu0 R)

        with p' fixed by the kinetic profiles and ff' derived from
        <j.B> = sigma_neo <E.B> + <j.B>_bootstrap.
        """
        # Locate axis and boundary for this psi iterate, following the
        # pattern used by the other Profile classes.
        self.eq._updateBoundaryPsi(psi)

        opt, xpt = critical.find_critical(R, Z, psi)
        if not opt:
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if psi_bndry is not None:
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif self.eq.psi_bndry is not None:
            psi_bndry = self.eq.psi_bndry
            mask = critical.core_mask(R, Z, psi, opt, xpt, psi_bndry)
        elif xpt:
            psi_bndry = xpt[0][2]
            mask = critical.core_mask(R, Z, psi, opt, xpt)
        else:
            psi_bndry = psi[0, 0]
            mask = None

        self.psi_axis = psi_axis
        self.psi_bndry = psi_bndry

        dpsi = psi_bndry - psi_axis
        if dpsi == 0.0:
            raise ValueError("psi_bndry equals psi_axis; no confined plasma")

        # --- 1D neoclassical calculation, and the 2D current components
        psi_norm = np.clip((psi - psi_axis) / dpsi, 0.0, 1.0)
        neo, Jtor_components = self._solve_1d(
            psi, psi_axis, psi_bndry, mask, psi_norm, opt, xpt
        )

        # --- Spline of the total ff' for the Profile interface
        self._ffprime_spline = interpolate.CubicSpline(
            self.psi_n, neo["ffprime"], extrapolate=True
        )
        self._ffprime_int = self._ffprime_spline.antiderivative()

        self.mask = mask
        self.Jtor_components = Jtor_components
        self.neo = neo

        return sum(Jtor_components.values())

    # ------------------------------------------------------------------
    # The neoclassical calculation
    # ------------------------------------------------------------------

    def _solve_1d(self, psi, psi_axis, psi_bndry, mask, psi_norm, opoint, xpoint):
        """
        Evaluate the neoclassical coefficients, ff' and the current components.

        Returns
        -------
        (neo, Jtor_components):
            ``neo`` is a dict of 1D arrays, all of length len(self.psi_n).
            ``Jtor_components`` is a dict of 2D (R, Z) current density arrays
            keyed "ohm", "bs" and "dia", which sum to the total Jtor.
        """
        x = self.psi_n

        # f = R*Btor from the current ff' iterate, not from the Equilibrium,
        # which does not yet hold this iterate's profile.
        neo = neoclassical_quantities(
            self.eq,
            self.kinetic,
            x,
            self._traced_slice,
            fpol=self.fpol(x),
            psi=psi,
            psi_axis=psi_axis,
            psi_bndry=psi_bndry,
            ntheta=self.ntheta,
            nray=self.nray,
            opoint=opoint,
            xpoint=xpoint,
            check_surfaces=self.check_surfaces,
        )
        self._surface_messages = neo["surface_messages"]

        fpol = neo["fpol"]
        B2_avg = neo["B2_avg"]
        pprime = neo["pprime"]
        jdotB_bs = neo["jdotB_bs"]
        jdotB_ohm_unit = neo["jdotB_ohm_unit"]

        # --- ff' from Eq. (*), split into components
        pre = mu0 * fpol / B2_avg
        ffprime_bs = pre * jdotB_bs
        ffprime_ohm_unit = pre * jdotB_ohm_unit
        ffprime_dia = -mu0 * fpol**2 * pprime / B2_avg

        # --- Map each component onto the (R, Z) grid.
        #
        # The diamagnetic piece uses the analytic p' rather than a spline of
        # it, exactly as pprime() does, so that the current integrated here is
        # the same current that is assembled and handed to the solver.
        J_bs = self._current_density(psi_norm, mask, ffprime_bs)
        J_ohm_unit = self._current_density(psi_norm, mask, ffprime_ohm_unit)
        J_dia = self._current_density(psi_norm, mask, ffprime_dia, include_pprime=True)

        # --- Choose Vloop so that the total current matches the target.
        #
        # Ip(Vloop) = I_dia + I_bs + Vloop * C, integrated with the same
        # Romberg quadrature that Equilibrium.solve uses, so that
        # eq.plasmaCurrent() reproduces the target to round-off.
        I_dia = self._integrate(J_dia)
        I_bs = self._integrate(J_bs)
        C = self._integrate(J_ohm_unit)

        if C == 0.0:
            raise ValueError(
                "Ohmic current response to loop voltage is zero; cannot "
                "constrain Ip. Check that the temperature profile gives a "
                "finite conductivity."
            )
        Vloop = (self.Ip - I_dia - I_bs) / C
        self.Vloop = Vloop

        ffprime_ohm = Vloop * ffprime_ohm_unit
        jdotB_ohm = Vloop * jdotB_ohm_unit

        Jtor_components = {
            "ohm": Vloop * J_ohm_unit,
            "bs": J_bs,
            "dia": J_dia,
        }

        neo.update(
            {
                "jdotB_ohm": jdotB_ohm,
                "jdotB_total": jdotB_bs + jdotB_ohm,
                "ffprime_bs": ffprime_bs,
                "ffprime_ohm": ffprime_ohm,
                "ffprime_dia": ffprime_dia,
                "ffprime": ffprime_bs + ffprime_ohm + ffprime_dia,
                "Vloop": Vloop,
                "I_dia": I_dia,
                "I_bs": I_bs,
                "I_ohm": Vloop * C,
            }
        )

        return neo, Jtor_components

    def _current_density(self, psi_norm, mask, ffprime_arr, include_pprime=False):
        """
        Toroidal current density of one component on the (R, Z) grid.

        J = ff'/(mu0 R), plus R p' for the diamagnetic component.
        """
        R = self.eq.R
        spl = interpolate.CubicSpline(self.psi_n, ffprime_arr, extrapolate=True)
        J = spl(psi_norm) / (mu0 * R)

        if include_pprime:
            J = J + R * self.pprime(psi_norm)

        if mask is not None:
            J = J * mask
        return J

    def _integrate(self, J):
        """
        Integrate a current density over the plasma cross-section, using the
        same Romberg quadrature as Equilibrium.solve.
        """
        dR = self.eq.R[1, 0] - self.eq.R[0, 0]
        dZ = self.eq.Z[0, 1] - self.eq.Z[0, 0]
        return float(romb(romb(J)) * dR * dZ)

    # ------------------------------------------------------------------
    # Profile interface
    # ------------------------------------------------------------------

    def pprime(self, psinorm):
        """dp/dpsi, fixed by the kinetic profiles."""
        if self.psi_axis is None:
            raise RuntimeError("pprime is only defined once Jtor has been called")
        pn = np.clip(psinorm, 0.0, 1.0)
        return self.kinetic.dpressure_dpsin(pn) / (self.psi_bndry - self.psi_axis)

    def ffprime(self, psinorm):
        """f df/dpsi from the neoclassical parallel current."""
        if self._ffprime_spline is None:
            # First iteration: no current profile yet, so f = fvac
            return np.zeros_like(np.asarray(psinorm, dtype=float))
        return self._ffprime_spline(np.clip(psinorm, 0.0, 1.0))

    def pressure(self, psinorm, out=None):
        """
        Total plasma pressure [Pa], taken directly from the kinetic profiles.

        Note this does not vanish at psinorm = 1 unless the supplied edge
        density or temperature does. That differs from the shape-function
        profile classes, which integrate p' subject to p(1) = 0.
        """
        result = self.kinetic.pressure(np.clip(psinorm, 0.0, 1.0))
        if out is not None:
            out[...] = result
            return out
        return result

    def fpol(self, psinorm, out=None):
        """
        f = R * Btor, from integrating ff' inward from the boundary.

        f^2(psi_n) = fvac^2 + 2 (psi_bndry - psi_axis) int_1^{psi_n} ff' dx
        """
        pn = np.clip(np.asarray(psinorm, dtype=float), 0.0, 1.0)

        if self._ffprime_spline is None:
            result = np.full(pn.shape, self._fvac)
        else:
            integral = self._ffprime_int(pn) - self._ffprime_int(1.0)
            f2 = self._fvac**2 + 2.0 * (self.psi_bndry - self.psi_axis) * integral
            if np.any(f2 <= 0.0):
                warnings.warn(
                    "f^2 went non-positive while integrating ff'; the "
                    "toroidal field is being driven through zero. Clipping.",
                    stacklevel=2,
                )
                f2 = np.maximum(f2, (0.01 * self._fvac) ** 2)
            result = np.sign(self._fvac) * np.sqrt(f2)

        if out is not None:
            out[...] = result
            return out
        return result

    def fvac(self):
        return self._fvac

    # ------------------------------------------------------------------

    def analysis(self):
        """
        Return a BootstrapAnalysis for the current state.

        Call after freegs.solve() has converged.
        """
        if not self.neo:
            raise RuntimeError("No solution yet; call freegs.solve() first")
        return BootstrapAnalysis(self)


class BootstrapAnalysis:
    """
    Post-run separation of the ohmic and bootstrap current contributions.

    Attributes
    ----------
    psi_n:
        Normalised flux grid of the 1D quantities
    traced:
        Boolean mask, True where the flux surface geometry was traced rather
        than frozen at the psi_max cutoff
    I_ohm, I_bs, I_dia:
        Ohmic, bootstrap and diamagnetic (Pfirsch-Schlueter) contributions to
        the total toroidal plasma current [Amps]. These sum to Ip.
    Ip:
        Total plasma current [Amps]
    f_bootstrap:
        I_bs / Ip
    Vloop:
        Loop voltage required to sustain Ip [V]
    Jtor_ohm, Jtor_bs, Jtor_dia:
        2D (R, Z) arrays of the current density components [A m^-2]
    """

    def __init__(self, profiles):
        self.profiles = profiles
        self.eq = profiles.eq
        neo = profiles.neo
        self.neo = neo

        self.psi_n = neo["psi_n"]
        self.traced = neo["traced"]
        self.Vloop = neo["Vloop"]

        self.I_ohm = neo["I_ohm"]
        self.I_bs = neo["I_bs"]
        self.I_dia = neo["I_dia"]
        self.Ip = self.I_ohm + self.I_bs + self.I_dia
        self.f_bootstrap = self.I_bs / self.Ip

        self.Jtor_ohm = profiles.Jtor_components["ohm"]
        self.Jtor_bs = profiles.Jtor_components["bs"]
        self.Jtor_dia = profiles.Jtor_components["dia"]

        # Radial current density profiles.
        #
        # dI = int Jtor dA_pol = dpsi * closed_int Jtor dl / (R |Bpol|), so
        # dI/dpsi = (dV/dpsi / 2pi) * <Jtor / R>, and for a component
        # <Jtor_X / R> = ff'_X <1/R^2> / mu0 (plus p' for the diamagnetic
        # piece).
        dpsi = profiles.psi_bndry - profiles.psi_axis
        pref = neo["dVdpsi"] / (2.0 * np.pi) * dpsi
        self.dIdpsin = {}
        for name in ("ohm", "bs", "dia"):
            integrand = neo[f"ffprime_{name}"] * neo["R2inv_avg"] / mu0
            if name == "dia":
                integrand = integrand + neo["pprime"]
            self.dIdpsin[name] = pref * integrand
        self.dIdpsin["total"] = sum(self.dIdpsin[n] for n in ("ohm", "bs", "dia"))

        # Parallel current densities, <j.B>/<B> is the usual reported form
        B_avg = np.sqrt(neo["B2_avg"])
        self.jpar_bs = neo["jdotB_bs"] / B_avg
        self.jpar_ohm = neo["jdotB_ohm"] / B_avg
        self.jpar_total = neo["jdotB_total"] / B_avg

    # ------------------------------------------------------------------

    def currents_from_radial_integral(self):
        """
        Recompute the component currents by integrating dI/dpsi_n.

        This is an independent path to I_ohm, I_bs and I_dia: it uses the
        flux surface geometry and 1D profiles rather than the 2D (R, Z)
        quadrature, so agreement between the two is a genuine consistency
        check on the flux surface averages.

        Returns a dict with keys "ohm", "bs", "dia", "total".
        """
        return {
            name: float(trapezoid(self.dIdpsin[name], self.psi_n))
            for name in ("ohm", "bs", "dia", "total")
        }

    def check_jdotB_consistency(self, psi_n=None):
        """
        Verify <j.B> reconstructed from p' and ff' against the Sauter value.

        Recomputes <j.B> = f p' + (f'/mu0) <B^2> from the converged ff' and
        p' and compares against the <j.B> that the Sauter formulas produced.
        These should agree to round-off: a mismatch means the ff' inversion
        or the flux surface averages are wrong.

        Returns (psi_n, relative_error).
        """
        neo = self.neo
        sel = self.traced if psi_n is None else np.isin(self.psi_n, psi_n)

        f = neo["fpol"][sel]
        ffp = neo["ffprime"][sel]
        pp = neo["pprime"][sel]
        B2 = neo["B2_avg"][sel]

        reconstructed = f * pp + (ffp / f) * B2 / mu0
        expected = neo["jdotB_total"][sel]

        scale = np.max(np.abs(expected))
        return self.psi_n[sel], (reconstructed - expected) / scale

    # ------------------------------------------------------------------

    def summary(self):
        """Return a formatted summary string."""
        neo = self.neo
        ip = self.Ip
        lines = ["Bootstrap current decomposition"]
        lines.append(
            f"  Total plasma current      Ip      = {ip:12.4e} A "
            f"(target {self.profiles.Ip:.4e} A)"
        )
        for label, value in (
            ("Ohmic                     I_ohm", self.I_ohm),
            ("Bootstrap                 I_bs ", self.I_bs),
            ("Diamagnetic (P-S)         I_dia", self.I_dia),
        ):
            lines.append(f"  {label}   = {value:12.4e} A ({100 * value / ip:6.2f}%)")
        lines.append(f"  Bootstrap fraction        f_bs    = {self.f_bootstrap:.4f}")
        lines.append(f"  Loop voltage              Vloop   = {self.Vloop:12.4e} V")

        radial = self.currents_from_radial_integral()
        lines.append("")
        lines.append("  Cross-check, radial integral of dI/dpsi_n:")
        for name, key in (
            ("ohmic", "ohm"),
            ("bootstrap", "bs"),
            ("diamagnetic", "dia"),
        ):
            direct = {"ohm": self.I_ohm, "bs": self.I_bs, "dia": self.I_dia}[key]
            rel = abs(radial[key] - direct) / max(abs(direct), 1.0)
            lines.append(
                f"    {name:<12s} {radial[key]:12.4e} A  vs 2D {direct:12.4e} A "
                f"(rel. diff {rel:.2e})"
            )

        _, err = self.check_jdotB_consistency()
        lines.append("")
        lines.append(
            f"  <j.B> reconstruction from p', ff': max rel. error "
            f"{np.max(np.abs(err)):.3e}"
        )

        i_cut = np.searchsorted(self.psi_n, self.profiles.psi_max)
        i_cut = min(i_cut, len(self.psi_n) - 1)
        lines.append("")
        lines.append(
            f"  Edge cutoff at psi_n = {self.profiles.psi_max:.4f}: "
            f"q = {neo['q'][i_cut]:.3f}, eps = {neo['eps'][i_cut]:.4f}, "
            f"f_t = {neo['ft'][i_cut]:.4f}, nu*_e = {neo['nue_star'][i_cut]:.3e}"
        )
        frozen = ~self.traced
        frozen[0] = False
        if np.any(frozen):
            frac = abs(
                float(np.trapezoid(self.dIdpsin["bs"][frozen], self.psi_n[frozen]))
                / max(abs(self.I_bs), 1.0)
            )
            lines.append(
                f"  Frozen-geometry region psi_n > {self.profiles.psi_max:.4f} "
                f"carries {100 * frac:.2f}% of I_bs"
            )
        return "\n".join(lines)

    def report(self):
        """Print the summary and a table of 1D neoclassical quantities."""
        print(self.summary())
        print()
        print(
            "  psi_n    f_t      q      eps    nu*_e     nu*_i     L31     "
            "L32     L34   alpha   sigma_neo   <j.B>_bs    <j.B>_ohm"
        )
        neo = self.neo
        step = max(1, len(self.psi_n) // 20)
        for i in range(0, len(self.psi_n), step):
            print(
                f"  {self.psi_n[i]:6.4f} {neo['ft'][i]:7.4f} {neo['q'][i]:7.3f} "
                f"{neo['eps'][i]:7.4f} {neo['nue_star'][i]:9.2e} "
                f"{neo['nui_star'][i]:9.2e} {neo['L31'][i]:7.3f} "
                f"{neo['L32'][i]:7.3f} {neo['L34'][i]:7.3f} "
                f"{neo['alpha'][i]:7.3f} {neo['sigma_neo'][i]:10.3e} "
                f"{neo['jdotB_bs'][i]:11.3e} {neo['jdotB_ohm'][i]:11.3e}"
            )

    def plot(self, axes=None, show=True):
        """Plot the current decomposition and neoclassical coefficients."""
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = np.asarray(axes).reshape(2, 3)
        neo = self.neo
        x = self.psi_n
        cut = self.profiles.psi_max

        ax = axes[0, 0]
        ax.plot(x, self.dIdpsin["total"], "k", label="total")
        ax.plot(x, self.dIdpsin["ohm"], label="ohmic")
        ax.plot(x, self.dIdpsin["bs"], label="bootstrap")
        ax.plot(x, self.dIdpsin["dia"], label="diamagnetic")
        ax.set_ylabel(r"$dI/d\psi_N$ [A]")
        ax.legend(fontsize="small")
        ax.set_title(f"$f_{{bs}}$ = {self.f_bootstrap:.3f}")

        ax = axes[0, 1]
        ax.plot(x, self.jpar_total, "k", label="total")
        ax.plot(x, self.jpar_ohm, label="ohmic")
        ax.plot(x, self.jpar_bs, label="bootstrap")
        ax.set_ylabel(r"$\langle j_\parallel B\rangle/\langle B\rangle$ [A m$^{-2}$]")
        ax.legend(fontsize="small")

        ax = axes[0, 2]
        ax.plot(x, neo["ft"], label="$f_t$")
        ax.plot(x, neo["eps"], label=r"$\epsilon$")
        ax.set_ylabel("trapped fraction / $\\epsilon$")
        ax.legend(fontsize="small")

        ax = axes[1, 0]
        ax.semilogy(x, neo["nue_star"], label=r"$\nu^*_e$")
        ax.semilogy(x, neo["nui_star"], label=r"$\nu^*_i$")
        ax.set_ylabel("collisionality")
        ax.legend(fontsize="small")

        ax = axes[1, 1]
        ax.plot(x, neo["L31"], label="$L_{31}$")
        ax.plot(x, neo["L32"], label="$L_{32}$")
        ax.plot(x, neo["L34"], label="$L_{34}$")
        ax.plot(x, neo["alpha"], label=r"$\alpha$")
        ax.set_ylabel("Sauter coefficients")
        ax.legend(fontsize="small")

        ax = axes[1, 2]
        ax.semilogy(x, neo["sigma_neo"], label=r"$\sigma_{neo}$")
        ax.set_ylabel(r"$\sigma$ [$\Omega^{-1}$m$^{-1}$]")
        ax.legend(fontsize="small")

        for ax in axes.flat:
            ax.axvline(cut, color="r", ls=":", lw=1)
            ax.set_xlabel(r"$\psi_N$")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        return axes


def analyse_equilibrium(
    eq,
    kinetic,
    npsi=64,
    ntheta=128,
    psi_min=0.02,
    psi_max=0.995,
    nray=256,
    check_surfaces=True,
):
    """
    Separate the bootstrap and ohmic currents of an *existing* equilibrium.

    Unlike BootstrapProfiles, this does not re-solve anything. It takes the
    equilibrium as given, which is what is wanted when validating against
    another code or an experimental reconstruction: the geometry and the total
    current profile are theirs, and only the bootstrap part is recomputed.

    The total parallel current is known exactly from the equilibrium, since

        <j.B> = f p' + (ff'/f) <B^2> / mu0

    with p' and ff' read from the equilibrium's own profiles. Sauter's formulas
    then supply <j.B>_bs from the kinetic profiles, and the ohmic current is
    what is left over:

        <j.B>_ohm = <j.B> - <j.B>_bs - (diamagnetic part)

    This is the standard experimental decomposition. Note it attributes
    everything that is not bootstrap or diamagnetic to the ohmic term, so any
    auxiliary current drive present in the equilibrium will be lumped in with
    the ohmic current.

    Parameters
    ----------
    eq:
        A solved Equilibrium, for example from freegs.geqdsk.read
    kinetic:
        KineticProfiles instance. These need not have generated the
        equilibrium; ``EquilibriumBootstrapAnalysis.pressure_mismatch``
        reports how well they agree with it.
    npsi, ntheta, psi_min, psi_max, nray, check_surfaces:
        As for BootstrapProfiles.

    Returns
    -------
    EquilibriumBootstrapAnalysis
    """
    psi_n, traced_slice = make_psin_grid(npsi, psi_min, psi_max)

    neo = neoclassical_quantities(
        eq,
        kinetic,
        psi_n,
        traced_slice,
        fpol=eq.fpol(psi_n),
        ntheta=ntheta,
        nray=nray,
        check_surfaces=check_surfaces,
    )
    return EquilibriumBootstrapAnalysis(eq, kinetic, neo, psi_max)


class EquilibriumBootstrapAnalysis:
    """
    Bootstrap / ohmic decomposition of a fixed equilibrium.

    Produced by analyse_equilibrium. See that function for what the
    decomposition means.

    Attributes
    ----------
    psi_n, traced:
        Normalised flux grid, and where the geometry was traced rather than
        frozen at the psi_max cutoff
    I_bs, I_ohm, I_dia, Ip:
        Component and total toroidal currents [Amps]
    f_bootstrap:
        I_bs / Ip
    Vloop_implied:
        Loop voltage implied by <j.B>_ohm = sigma_neo <E.B> at each radius [V].
        In a stationary discharge this is radially constant; how much it varies
        indicates how far the current profile is from relaxed, or how large the
        non-inductive current not accounted for here is.
    pressure_mismatch:
        max |p_kinetic - p_equilibrium| / max(p_equilibrium), a check that the
        supplied profiles actually describe this equilibrium
    """

    def __init__(self, eq, kinetic, neo, psi_max):
        self.eq = eq
        self.kinetic = kinetic
        self.neo = neo
        self.psi_max = psi_max
        self.psi_n = neo["psi_n"]
        self.traced = neo["traced"]

        x = self.psi_n
        dpsi = eq.psi_bndry - eq.psi_axis
        f = neo["fpol"]
        B2 = neo["B2_avg"]

        # --- Total <j.B> from the equilibrium's own p' and ff'
        pprime_eq = np.asarray(eq.pprime(x), dtype=float)
        ffprime_eq = np.asarray(eq.ffprime(x), dtype=float)
        self.pprime_eq = pprime_eq
        self.ffprime_eq = ffprime_eq

        jdotB_total = f * pprime_eq + (ffprime_eq / f) * B2 / mu0

        # --- Component split, exactly additive by construction
        ffprime_dia = -mu0 * f**2 * pprime_eq / B2
        ffprime_bs = mu0 * f * neo["jdotB_bs"] / B2
        ffprime_ohm = ffprime_eq - ffprime_dia - ffprime_bs

        jdotB_bs = neo["jdotB_bs"]
        jdotB_ohm = (
            jdotB_total - jdotB_bs - f * pprime_eq - (ffprime_dia / f) * B2 / mu0
        )

        self.jdotB_total = jdotB_total
        self.jdotB_bs = jdotB_bs
        self.jdotB_ohm = jdotB_ohm

        # --- Radial current profiles, dI/dpsi_n = (dV/dpsi / 2pi) <Jtor/R> dpsi
        pref = neo["dVdpsi"] / (2.0 * np.pi) * dpsi
        self.dIdpsin = {}
        for name, ffp in (
            ("ohm", ffprime_ohm),
            ("bs", ffprime_bs),
            ("dia", ffprime_dia),
        ):
            integrand = ffp * neo["R2inv_avg"] / mu0
            if name == "dia":
                integrand = integrand + pprime_eq
            self.dIdpsin[name] = pref * integrand
        self.dIdpsin["total"] = sum(self.dIdpsin[n] for n in ("ohm", "bs", "dia"))

        self.ffprime_components = {
            "ohm": ffprime_ohm,
            "bs": ffprime_bs,
            "dia": ffprime_dia,
        }

        self.I_ohm = float(trapezoid(self.dIdpsin["ohm"], x))
        self.I_bs = float(trapezoid(self.dIdpsin["bs"], x))
        self.I_dia = float(trapezoid(self.dIdpsin["dia"], x))
        self.Ip = self.I_ohm + self.I_bs + self.I_dia
        self.f_bootstrap = self.I_bs / self.Ip

        # Current from the equilibrium itself, as an accuracy check on the
        # flux surface quadrature
        self.Ip_equilibrium = float(eq.plasmaCurrent())

        # --- Implied loop voltage, from <j.B>_ohm = sigma_neo <E.B>
        with np.errstate(divide="ignore", invalid="ignore"):
            self.Vloop_implied = np.where(
                neo["sigma_neo"] > 0.0,
                2.0 * np.pi * jdotB_ohm / (neo["sigma_neo"] * f * neo["R2inv_avg"]),
                np.nan,
            )

        # --- Consistency of the supplied profiles with the equilibrium
        p_kin = kinetic.pressure(x)
        p_eq = np.asarray(eq.pressure(x), dtype=float)
        self.p_kinetic = p_kin
        self.p_equilibrium = p_eq
        scale = np.max(np.abs(p_eq))
        self.pressure_mismatch = (
            float(np.max(np.abs(p_kin - p_eq)) / scale) if scale > 0 else np.inf
        )

    # ------------------------------------------------------------------

    def summary(self):
        """Return a formatted summary string."""
        neo = self.neo
        ip = self.Ip
        lines = ["Bootstrap decomposition of a fixed equilibrium"]
        lines.append(
            f"  Total plasma current      Ip      = {ip:12.4e} A  "
            f"(equilibrium: {self.Ip_equilibrium:.4e} A, "
            f"rel. diff {abs(ip - self.Ip_equilibrium) / abs(self.Ip_equilibrium):.2e})"
        )
        for label, value in (
            ("Ohmic + other non-BS  I_ohm", self.I_ohm),
            ("Bootstrap             I_bs ", self.I_bs),
            ("Diamagnetic (P-S)     I_dia", self.I_dia),
        ):
            lines.append(f"  {label}   = {value:12.4e} A ({100 * value / ip:6.2f}%)")
        lines.append(f"  Bootstrap fraction        f_bs    = {self.f_bootstrap:.4f}")

        # Implied loop voltage over the region where it is meaningful
        sel = self.traced & np.isfinite(self.Vloop_implied)
        if np.any(sel):
            v = self.Vloop_implied[sel]
            lines.append(
                f"  Implied Vloop             range   = [{np.min(v):.3f}, "
                f"{np.max(v):.3f}] V, median {np.median(v):.3f} V"
            )
            lines.append(
                "    (a wide range means the current profile is not relaxed, or "
                "there is non-inductive drive not modelled here)"
            )

        lines.append("")
        lines.append(
            f"  Kinetic vs equilibrium pressure: max rel. difference "
            f"{100 * self.pressure_mismatch:.2f}%"
        )
        if self.pressure_mismatch > 0.1:
            lines.append(
                "    WARNING: the supplied n and T profiles do not reproduce "
                "this equilibrium's pressure. The bootstrap estimate is only "
                "as good as that agreement."
            )

        i_cut = min(int(np.searchsorted(self.psi_n, self.psi_max)), len(self.psi_n) - 1)
        lines.append("")
        lines.append(
            f"  Edge cutoff at psi_n = {self.psi_max:.4f}: "
            f"q = {neo['q'][i_cut]:.3f}, eps = {neo['eps'][i_cut]:.4f}, "
            f"f_t = {neo['ft'][i_cut]:.4f}, nu*_e = {neo['nue_star'][i_cut]:.3e}"
        )
        return "\n".join(lines)

    def report(self):
        """Print the summary and a table of 1D neoclassical quantities."""
        print(self.summary())
        print()
        print(
            "  psi_n    f_t      q      eps    nu*_e     nu*_i     L31     "
            "L32     L34   alpha   sigma_neo   <j.B>_bs   <j.B>_tot"
        )
        neo = self.neo
        step = max(1, len(self.psi_n) // 20)
        for i in range(0, len(self.psi_n), step):
            print(
                f"  {self.psi_n[i]:6.4f} {neo['ft'][i]:7.4f} {neo['q'][i]:7.3f} "
                f"{neo['eps'][i]:7.4f} {neo['nue_star'][i]:9.2e} "
                f"{neo['nui_star'][i]:9.2e} {neo['L31'][i]:7.3f} "
                f"{neo['L32'][i]:7.3f} {neo['L34'][i]:7.3f} "
                f"{neo['alpha'][i]:7.3f} {neo['sigma_neo'][i]:10.3e} "
                f"{self.jdotB_bs[i]:10.3e} {self.jdotB_total[i]:10.3e}"
            )

    def plot(self, axes=None, show=True):
        """Plot the decomposition, coefficients and consistency checks."""
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = np.asarray(axes).reshape(2, 3)
        neo = self.neo
        x = self.psi_n

        ax = axes[0, 0]
        ax.plot(x, self.dIdpsin["total"], "k", label="total")
        ax.plot(x, self.dIdpsin["ohm"], label="ohmic + other")
        ax.plot(x, self.dIdpsin["bs"], label="bootstrap")
        ax.plot(x, self.dIdpsin["dia"], label="diamagnetic")
        ax.set_ylabel(r"$dI/d\psi_N$ [A]")
        ax.set_title(f"$f_{{bs}}$ = {self.f_bootstrap:.3f}")
        ax.legend(fontsize="small")

        ax = axes[0, 1]
        ax.plot(x, self.jdotB_total, "k", label="total")
        ax.plot(x, self.jdotB_ohm, label="ohmic + other")
        ax.plot(x, self.jdotB_bs, label="bootstrap")
        ax.set_ylabel(r"$\langle j_\parallel B\rangle$ [A T m$^{-2}$]")
        ax.legend(fontsize="small")

        ax = axes[0, 2]
        ax.plot(x, self.p_equilibrium, "k", label="equilibrium")
        ax.plot(x, self.p_kinetic, "--", label="from $n$, $T$")
        ax.set_ylabel("pressure [Pa]")
        ax.set_title(f"mismatch {100 * self.pressure_mismatch:.1f}%")
        ax.legend(fontsize="small")

        ax = axes[1, 0]
        ax.plot(x, neo["ft"], label="$f_t$")
        ax.plot(x, neo["eps"], label=r"$\epsilon$")
        ax.set_ylabel("trapped fraction / $\\epsilon$")
        ax.legend(fontsize="small")

        ax = axes[1, 1]
        ax.semilogy(x, neo["nue_star"], label=r"$\nu^*_e$")
        ax.semilogy(x, neo["nui_star"], label=r"$\nu^*_i$")
        ax.set_ylabel("collisionality")
        ax.legend(fontsize="small")

        ax = axes[1, 2]
        ax.plot(x, self.Vloop_implied)
        ax.set_ylabel("implied $V_{loop}$ [V]")
        ax.set_ylim(
            np.nanpercentile(self.Vloop_implied, 2),
            np.nanpercentile(self.Vloop_implied, 98),
        )

        for ax in axes.flat:
            ax.axvline(self.psi_max, color="r", ls=":", lw=1)
            ax.set_xlabel(r"$\psi_N$")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        return axes
