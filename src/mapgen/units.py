"""The three unit conversions the map-generation code needs.

These were previously taken from `constant.py`, a 451-line module by Andrew Wetzel and
Andrew Emerick that ships with their `utilities` / `gizmo_analysis` packages. Only these
three values were ever used, so the vendored copy is gone rather than republished here --
see docs/THIRD_PARTY.md. The expressions below are the upstream ones, and the release test
pins the resulting floats so a refactor cannot quietly move them.

If you need more of that module, install it upstream rather than copying it back in:
    https://bitbucket.org/awetzel/utilities  (also a dependency of gizmo_analysis)
"""

from scipy import constants as scipy_const

# 1 pc in cm, then 1 kpc in km: parsec[cm] x 1e-2 gives km/pc, x1e3 for kpc -- the upstream
# chain is cm_per_pc * 1e-2, which is km per kpc because the two factors of 1e3 cancel.
_cm_per_pc = scipy_const.parsec * 1e2
km_per_kpc = _cm_per_pc * 1e-2

# Julian year, not tropical: 365.25 days exactly.
_sec_per_yr = scipy_const.Julian_year
sec_per_Gyr = _sec_per_yr * 1e9

# Written as the reciprocal of degrees-per-radian, matching upstream: pi/180 computed the
# other way round can differ in the last bit.
_degree_per_radian = 180 / scipy_const.pi
radian_per_degree = 1 / _degree_per_radian
