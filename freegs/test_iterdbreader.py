"""
Test for iterdb_reader.py

This script checks that density and temperature profiles are imported correctly.

The checks run against the real DIII-D ITERDB files in the ``iterdb/``
directory at the top of the repository. Those files are data, not code, so
every test is skipped if the directory is absent.

Run as a module

    python -m freegs.test_iterdbreader

it parses each ITERDB file and shows the density, temperature and pressure
profiles instead of running the assertions.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from freegs import iterdb_reader
from freegs.sauter import ELEMENTARY_CHARGE

# ----------------------------------------------------------------------------
# Test data
# ----------------------------------------------------------------------------

ITERDB_DIR = Path(__file__).resolve().parent.parent / "iterdb"
ITERDB_FILES = sorted(ITERDB_DIR.glob("*.iterdb")) if ITERDB_DIR.is_dir() else []

# Every ITERDB file here is written by write_iterdb.py for GENE, so all six
# profiles below are expected in each of them.
EXPECTED_PROFILES = ["TE", "TI", "NE", "NM1", "NM2", "VROT"]

# NM2 is the carbon impurity in these DIII-D cases; ne = n_C6+ * 6 + n_D+
IMPURITY_Z = 6.0

pytestmark = pytest.mark.skipif(
    not ITERDB_FILES, reason=f"no ITERDB files found in {ITERDB_DIR}"
)

# Parsing 4000-point profiles with np.append is not free, so read each file
# once and share it between tests.
_cache = {}


def load(filename):
    """Parse an ITERDB file, reusing an earlier parse of the same file."""
    key = str(filename)
    if key not in _cache:
        _cache[key] = iterdb_reader.ITERDBReader(key)
    return _cache[key]


def file_ids(paths):
    return [p.stem for p in paths]


# ----------------------------------------------------------------------------
# An independent parser, used to check the fixed-width column reader
# ----------------------------------------------------------------------------


_FLOAT_RE = re.compile(r"[-+]?\d+\.\d+e[-+]?\d+")


def _data_line_values(line):
    """
    Pull every exponential-notation float out of a line, or None if the line
    is not pure data.

    Regex matching rather than a whitespace split, because negative values are
    13 characters wide and so run into the preceding column with no separator
    (see the VROT sections). Returning None for anything that is not entirely
    floats skips the header lines and the single time-value line that sits
    between the independent and dependent blocks of each section.
    """
    values = _FLOAT_RE.findall(line)
    if not values:
        return None
    if _FLOAT_RE.sub("", line).strip():
        return None  # something other than floats on the line
    return [float(v) for v in values]


def reparse_independently(filename):
    """
    Re-read an ITERDB file by scanning each line for float literals.

    ITERDBReader slices fixed 13-character columns instead, so this gives an
    independent answer to compare against.

    Returns
    -------
    tuple
        (num_points, {quantity: (indep, dep)})
    """
    lines = Path(filename).read_text().split("\n")

    num_points = None
    for line in lines:
        if "# OF X PTS" in line:
            num_points = int(line.split()[0])
            break
    assert num_points is not None, "no '# OF X PTS' header line"

    profiles = {}
    quantity = None
    i = 0
    while i < len(lines):
        if "-DEPENDENT VARIABLE LABEL" in lines[i]:
            quantity = lines[i].split()[0]
        elif "DATA FOLLOW" in lines[i] and quantity is not None:
            values = []
            i += 1
            while i < len(lines) and len(values) < 2 * num_points:
                found = _data_line_values(lines[i])
                if found is not None:
                    values.extend(found)
                i += 1
            assert len(values) == 2 * num_points, (
                f"{quantity}: got {len(values)} values, expected {2 * num_points}"
            )
            profiles[quantity] = (
                np.array(values[:num_points]),
                np.array(values[num_points:]),
            )
            quantity = None
            continue
        i += 1

    return num_points, profiles


# ----------------------------------------------------------------------------
# Importing
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_all_expected_profiles_are_read(filename):
    reader = load(filename)
    assert reader.list_profiles() == EXPECTED_PROFILES


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_every_profile_has_the_number_of_points_in_the_header(filename):
    num_points, _ = reparse_independently(filename)
    reader = load(filename)
    for quantity in reader.list_profiles():
        info = reader.get_profile_info(quantity)
        assert info["num_points"] == num_points
        assert len(reader.profiles[quantity]["data"]) == num_points


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_fixed_width_parse_agrees_with_independent_parse(filename):
    """
    The reader slices fixed 13-character columns. Compare every value against
    a regex scan of the same file, which is where a silent off-by-one in the
    column arithmetic would show up.
    """
    _, expected = reparse_independently(filename)
    reader = load(filename)

    assert set(reader.list_profiles()) == set(expected)
    for quantity, (indep, dep) in expected.items():
        stored = reader.profiles[quantity]
        assert np.array_equal(stored["rhot"], indep), f"{quantity} rho_tor differs"
        assert np.array_equal(stored["data"], dep), f"{quantity} data differs"


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_radial_grid_is_shared_increasing_and_normalised(filename):
    reader = load(filename)
    rhot = reader.profiles["TE"]["rhot"]

    assert np.all(np.diff(rhot) > 0.0), "rho_tor is not strictly increasing"
    assert 0.0 <= rhot[0] < 0.01
    # The grid runs to the separatrix; DIIID199091 overshoots it very slightly
    assert 0.99 < rhot[-1] < 1.01

    # All quantities in an ITERDB file sit on the same grid; anything else
    # means a section boundary was misread.
    for quantity in reader.list_profiles():
        assert np.array_equal(reader.profiles[quantity]["rhot"], rhot), quantity


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_units_are_read_from_the_label_line(filename):
    reader = load(filename)
    assert reader.get_profile_info("TE")["units"] == "eV"
    assert reader.get_profile_info("TI")["units"] == "eV"
    for quantity in ("NE", "NM1", "NM2"):
        assert reader.get_profile_info(quantity)["units"] == "m^-3"
    assert reader.get_profile_info("VROT")["units"] == "rad/s"


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_missing_file_raises(filename):
    with pytest.raises(FileNotFoundError):
        iterdb_reader.ITERDBReader(str(filename) + ".does-not-exist")


def test_unknown_quantity_raises():
    reader = load(ITERDB_FILES[0])
    with pytest.raises(ValueError, match="not found"):
        reader.get_profile("NOT_A_QUANTITY")
    with pytest.raises(ValueError, match="not found"):
        reader.get_profile_info("NOT_A_QUANTITY")


# ----------------------------------------------------------------------------
# Physical sanity of what was imported
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_densities_and_temperatures_are_finite_and_positive(filename):
    reader = load(filename)
    for quantity in ("TE", "TI", "NE", "NM1", "NM2"):
        data = reader.profiles[quantity]["data"]
        assert np.all(np.isfinite(data)), quantity
        assert np.all(data > 0.0), quantity


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_magnitudes_are_in_the_expected_units(filename):
    """
    Catch a factor-of-1000 slip in temperature (eV read as keV) or a missing
    1e19 in density by checking the on-axis values against DIII-D ranges.
    """
    reader = load(filename)

    Te0 = reader.profiles["TE"]["data"][0]
    Ti0 = reader.profiles["TI"]["data"][0]
    assert 500.0 < Te0 < 3.0e4, f"Te(0) = {Te0} eV is not a DIII-D core value"
    assert 500.0 < Ti0 < 3.0e4, f"Ti(0) = {Ti0} eV is not a DIII-D core value"

    ne0 = reader.profiles["NE"]["data"][0]
    assert 1.0e19 < ne0 < 1.0e21, f"ne(0) = {ne0} m^-3 is not a DIII-D core value"


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_temperature_falls_from_axis_to_edge(filename):
    """
    Te and Ti must decrease outwards. Checked on a coarse subsample so that
    experimental wiggle in the raw data is not mistaken for a parsing error.
    """
    reader = load(filename)
    coarse = np.linspace(0, len(reader.profiles["TE"]["rhot"]) - 1, 21).astype(int)

    for quantity in ("TE", "TI"):
        data = reader.profiles[quantity]["data"][coarse]
        assert np.all(np.diff(data) < 0.0), f"{quantity} is not decreasing outwards"
        assert data[0] > 3.0 * data[-1], f"{quantity} has no core-to-edge contrast"

    # Density need not be monotonic (these cases can be slightly hollow), but
    # the edge must still be well below the core.
    ne = reader.profiles["NE"]["data"]
    assert ne[0] > 2.0 * ne[-1]


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_quasineutrality_holds_with_carbon(filename):
    """
    ne = n_D + 6 n_C at every radius. This ties the three density profiles
    together, so it fails if any one of them is misaligned by even a single
    grid point.
    """
    reader = load(filename)
    ne = reader.profiles["NE"]["data"]
    ni = reader.profiles["NM1"]["data"]
    nimp = reader.profiles["NM2"]["data"]

    assert np.allclose(ne, ni + IMPURITY_Z * nimp, rtol=1e-5)


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_effective_charge_is_physical(filename):
    """Zeff from the ion densities must lie between 1 and the impurity Z."""
    reader = load(filename)
    ne = reader.profiles["NE"]["data"]
    ni = reader.profiles["NM1"]["data"]
    nimp = reader.profiles["NM2"]["data"]

    Zeff = (ni + IMPURITY_Z**2 * nimp) / ne
    assert np.all(Zeff > 1.0)
    assert np.all(Zeff < IMPURITY_Z)


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_pressure_is_positive_and_centrally_peaked(filename):
    reader = load(filename)
    rhot = reader.profiles["NE"]["rhot"]
    p = total_pressure(reader, rhot)

    assert np.all(p > 0.0)
    # Peaked in the core. Not necessarily exactly on axis: DIIID171322 is
    # mildly hollow inside rho_tor ~ 0.16 because its density is.
    assert rhot[np.argmax(p)] < 0.25, "pressure does not peak in the core"
    assert p[0] > 0.95 * np.max(p)
    # Tens of kPa on axis, falling to order 1 kPa at the separatrix
    assert 1.0e4 < p[0] < 1.0e6
    assert p[0] > 10.0 * p[-1]


# ----------------------------------------------------------------------------
# Interpolated getters
# ----------------------------------------------------------------------------

GETTERS = ["get_Te", "get_Ti", "get_ne", "get_ni"]
GETTER_QUANTITY = {"get_Te": "TE", "get_Ti": "TI", "get_ne": "NE", "get_ni": "NM1"}


@pytest.mark.parametrize("getter", GETTERS)
@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_getter_returns_a_callable_when_given_no_grid(filename, getter):
    reader = load(filename)
    f = getattr(reader, getter)()
    assert callable(f)

    x = np.linspace(0.0, 1.0, 11)
    assert np.allclose(f(x), getattr(reader, getter)(x))


@pytest.mark.parametrize("getter", GETTERS)
@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_getter_reproduces_the_raw_data_at_the_grid_points(filename, getter):
    """Interpolation must pass through the data it was built from."""
    reader = load(filename)
    quantity = GETTER_QUANTITY[getter]
    rhot = reader.profiles[quantity]["rhot"]
    data = reader.profiles[quantity]["data"]

    nodes = np.linspace(0, len(rhot) - 1, 50).astype(int)
    values = getattr(reader, getter)(rhot[nodes])
    assert np.allclose(values, data[nodes], rtol=1e-8)


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_getters_stay_positive_between_and_beyond_the_grid_points(filename):
    reader = load(filename)
    x = np.linspace(0.0, 1.0, 500)
    for getter in GETTERS:
        values = getattr(reader, getter)(x)
        assert np.all(np.isfinite(values)), getter
        assert np.all(values > 0.0), getter


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_get_profile_matches_the_named_getters(filename):
    reader = load(filename)
    x = np.linspace(0.05, 0.95, 33)
    for getter, quantity in GETTER_QUANTITY.items():
        assert np.allclose(
            reader.get_profile(quantity, x), getattr(reader, getter)(x), rtol=1e-12
        )


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_get_profile_info_reports_the_stored_arrays(filename):
    reader = load(filename)
    for quantity in reader.list_profiles():
        info = reader.get_profile_info(quantity)
        stored = reader.profiles[quantity]

        assert info["quantity"] == quantity
        assert info["units"] == stored["units"]
        assert info["rhot_range"] == (stored["rhot"].min(), stored["rhot"].max())
        assert info["data_range"] == (stored["data"].min(), stored["data"].max())


# ----------------------------------------------------------------------------
# Unit conversion, which the eV / m^-3 files above never exercise
# ----------------------------------------------------------------------------


def write_synthetic_iterdb(path, quantity, units, values, rhot):
    """
    Write a single-profile ITERDB file in the fixed-width layout the reader
    expects: six 13-character columns per line, the independent variable
    first, then a time line, then the dependent variable.
    """
    num_points = len(rhot)

    def block(array):
        lines = []
        for start in range(0, num_points, 6):
            row = array[start : start + 6]
            lines.append(" " + "".join(f" {v:12.6e}" for v in row))
        return lines

    lines = [
        ";Synthetic ITERDB file for test_iterdbreader.py",
        ";----END-OF-ORIGINAL-HEADER------COMMENTS:-----------",
        " 99999  xyz 2              ;-SHOT #- F(X) DATA ",
        "   0                          ;-NUMBER OF ASSOCIATED SCALAR QUANTITIES",
        " RHOTOR              -        ;-INDEPENDENT VARIABLE LABEL: X-",
        " TIME                SECONDS  ;-INDEPENDENT VARIABLE LABEL: Y-",
        f" {quantity:<19s} {units:<12s} ;-DEPENDENT VARIABLE LABEL",
        " 3                            ;-PROC CODE- 0:RAW 1:AVG 2:SM. 3:AVG+SM",
        f"      {num_points}                   ;-# OF X PTS- ",
        "      1                   ;-# OF Y PTS-  X,Y,F(X,Y) DATA FOLLOW:",
    ]
    lines += block(rhot)
    lines += ["  01359"]
    lines += block(values)
    lines += [";----END-OF-DATA-----------------COMMENTS:-----------"]

    Path(path).write_text("\n".join(lines) + "\n")


def test_temperature_in_keV_is_converted_to_eV(tmp_path):
    rhot = np.linspace(0.0, 1.0, 25)
    Te_keV = 3.0 - 2.9 * rhot**2

    path = tmp_path / "keV.iterdb"
    write_synthetic_iterdb(path, "TE", "keV", Te_keV, rhot)
    reader = iterdb_reader.ITERDBReader(str(path))

    # Stored in file units, returned in eV
    assert np.allclose(reader.profiles["TE"]["data"], Te_keV)
    assert np.allclose(reader.get_Te(rhot), Te_keV * 1000.0, rtol=1e-6)


def test_density_in_1E19_units_is_converted(tmp_path):
    rhot = np.linspace(0.0, 1.0, 25)
    ne_19 = 8.0 - 6.0 * rhot**2

    path = tmp_path / "n19.iterdb"
    write_synthetic_iterdb(path, "NE", "1E19m^-3", ne_19, rhot)
    reader = iterdb_reader.ITERDBReader(str(path))

    assert np.allclose(reader.get_ne(rhot), ne_19 * 1e19, rtol=1e-6)


def test_optional_profiles_return_None_when_absent(tmp_path):
    """TI and NM1 are optional; the reader says so instead of raising."""
    rhot = np.linspace(0.0, 1.0, 25)
    path = tmp_path / "te-only.iterdb"
    write_synthetic_iterdb(path, "TE", "eV", 3000.0 - 2900.0 * rhot**2, rhot)
    reader = iterdb_reader.ITERDBReader(str(path))

    assert reader.get_Ti() is None
    assert reader.get_ni() is None
    with pytest.raises(ValueError, match="NE profile not found"):
        reader.get_ne()


def test_header_without_point_count_is_rejected(tmp_path):
    path = tmp_path / "broken.iterdb"
    path.write_text(";not an ITERDB file\n" + "\n".join([" 0.0"] * 20) + "\n")
    with pytest.raises(ValueError, match="Could not find number of points"):
        iterdb_reader.ITERDBReader(str(path))


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------


def total_pressure(reader, rhot):
    """
    Thermal pressure P = n T summed over species, in Pascals.

    ITERDB temperatures are in eV, so the electron charge converts n[m^-3]
    times T[eV] into Pa.
    """
    pe = reader.get_ne(rhot) * reader.get_Te(rhot)

    Ti = reader.get_Ti(rhot)
    if Ti is None:
        Ti = reader.get_Te(rhot)

    ni = reader.get_ni(rhot)
    if ni is None:
        ni = reader.get_ne(rhot)
    if "NM2" in reader.profiles:
        ni = ni + reader.get_profile("NM2", rhot)

    return ELEMENTARY_CHARGE * (pe + ni * Ti)


def plot_profiles(reader, npoints=400, axes=None):
    """
    Plot density, temperature and pressure against rho_tor.

    Parameters
    ----------
    reader : ITERDBReader
        Reader holding the profiles to plot.
    npoints : int, optional
        Number of points at which to evaluate the interpolated profiles.
    axes : array of matplotlib Axes, optional
        Three axes to draw into. Created if not given.

    Returns
    -------
    tuple
        (figure, axes)
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 9))
    axes = np.atleast_1d(axes).ravel()
    assert len(axes) >= 3, "plot_profiles needs three axes"

    rhot = np.linspace(0.0, 1.0, npoints)
    nodes = reader.profiles["NE"]["rhot"]

    # Density
    ax = axes[0]
    ax.plot(rhot, reader.get_ne(rhot) / 1e19, label=r"$n_e$")
    ni = reader.get_ni(rhot)
    if ni is not None:
        ax.plot(rhot, ni / 1e19, "--", label=r"$n_i$ (NM1)")
    if "NM2" in reader.profiles:
        ax.plot(
            rhot, reader.get_profile("NM2", rhot) / 1e19, ":", label=r"$n_{imp}$ (NM2)"
        )
    ax.set_ylabel(r"$n$  [$10^{19}\,\mathrm{m^{-3}}$]")
    ax.set_title(f"Density, temperature and pressure\n{Path(reader.filename).name}")

    # Temperature
    ax = axes[1]
    ax.plot(rhot, reader.get_Te(rhot) / 1e3, label=r"$T_e$")
    Ti = reader.get_Ti(rhot)
    if Ti is not None:
        ax.plot(rhot, Ti / 1e3, "--", label=r"$T_i$")
    ax.set_ylabel(r"$T$  [keV]")

    # Pressure, P = nT
    ax = axes[2]
    ax.plot(rhot, total_pressure(reader, rhot) / 1e3, "k", label=r"$P = \sum nT$")
    ax.plot(
        rhot,
        ELEMENTARY_CHARGE * reader.get_ne(rhot) * reader.get_Te(rhot) / 1e3,
        "--",
        label=r"$P_e = n_e T_e$",
    )
    ax.set_ylabel(r"$P$  [kPa]")
    ax.set_xlabel(r"$\rho_{tor}$")

    for ax in axes[:3]:
        # Mark the extent of the data so extrapolation is visible
        ax.axvspan(0.0, nodes[0], color="0.9")
        ax.axvspan(nodes[-1], 1.0, color="0.9")
        ax.set_xlim(0.0, 1.0)
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    return axes[0].figure, axes


@pytest.mark.parametrize("filename", ITERDB_FILES, ids=file_ids(ITERDB_FILES))
def test_plot_profiles_draws_three_populated_panels(filename):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reader = load(filename)
    fig, axes = plot_profiles(reader, npoints=50)
    try:
        assert len(axes) == 3
        for ax in axes:
            assert ax.lines, "empty panel"
            for line in ax.lines:
                y = line.get_ydata()
                assert np.all(np.isfinite(y)), line.get_label()
                assert np.all(y > 0.0), line.get_label()
    finally:
        plt.close(fig)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    if not ITERDB_FILES:
        raise SystemExit(f"No ITERDB files found in {ITERDB_DIR}")

    for filename in ITERDB_FILES:
        reader = load(filename)
        print(f"\n{filename.name}")
        for quantity in reader.list_profiles():
            info = reader.get_profile_info(quantity)
            print(
                f"  {quantity:5s} [{info['units']:8s}] "
                f"{info['num_points']:5d} points, "
                f"rho_tor {info['rhot_range'][0]:.4f}..{info['rhot_range'][1]:.4f}, "
                f"data {info['data_range'][0]:.4e}..{info['data_range'][1]:.4e}"
            )
        fig, _ = plot_profiles(reader)
        fig.tight_layout()

    plt.show()
