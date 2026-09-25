import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start):
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
