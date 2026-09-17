"""Set-level (text-to-image) diversity comparison on FLUX rectified flow.

Four arms over identical starting latents -- the unmodified sampler, OSCAR,
JIVE, and JIVE+CADS -- plus the standalone CADS baseline in
baselines/cads.py. See main.py for the full arm descriptions and the CLI.

Run as `python -m jive_flux ...` from the set_level/ directory. The JIVE
repository root (which holds the shared `core` and `baselines` packages) is
located by walking upward from here, so the imports work regardless of the
caller's cwd.
"""
import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start):
    """Walk upward from `start` to locate the JIVE repository root
    (identified by core/vendi.py and baselines/oscar/utils.py living inside
    it)."""
    d = os.path.abspath(start)
    while True:
        if (os.path.isfile(os.path.join(d, "core", "vendi.py"))
                and os.path.isfile(os.path.join(d, "baselines", "oscar", "utils.py"))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(
                "Could not locate the JIVE repository root (core/ + baselines/) "
                "in any directory above " + start
            )
        d = parent


REPO_ROOT = _find_repo_root(_PKG_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
