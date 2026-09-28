"""Project paths and import roots for the three-hinge study. Every script in this package starts
by importing this module; nothing else in the package spells out an absolute path.

By default the project data is looked for beside this repository. To put it elsewhere, set the
environment variables:

    EXVIVO_ROOT         project root; holds surf/ (FreeSurfer subjects, surface maps, GyralNet
                        output, tractography, results) and the ex-vivo dataset 28211843/
    EXVIVO_AABC         AABC Release 2 root (organized/ and derived/), for the in-vivo scripts
    EXVIVO_AABC_EXAMPLE the AABC session shown as the in-vivo example in Fig. 4a, for instance
                        the name of one directory under derived/. It has no default: AABC is
                        distributed under a data use agreement, and participant identifiers
                        are not written into this repository.
    FREESURFER_HOME     FreeSurfer 7.x

Data stays where it is - under EXVIVO_ROOT - and only the CODE lives here. In particular the
tractography and results directories under EXVIVO_ROOT/surf/three_hinge/ are data directories
that happen to share a name with the folders the scripts were developed in."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Defaults are relative to the directory that holds this repository: the project data is expected
# beside it, and AABC inside the project directory (a symbolic link will do).
ROOT = os.environ.get("EXVIVO_ROOT", os.path.abspath(os.path.join(HERE, "..", "..")))
AABC = os.environ.get("EXVIVO_AABC", os.path.join(ROOT, "AABC_R2"))
FREESURFER_HOME = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")
AABC_EXAMPLE = os.environ.get("EXVIVO_AABC_EXAMPLE", "")


def aabc_example():
    """The session shown in Fig. 4a; asks for the variable rather than guessing a participant."""
    if not AABC_EXAMPLE:
        raise SystemExit("Set EXVIVO_AABC_EXAMPLE to the AABC session to show in Fig. 4a "
                         "(the name of one directory under $EXVIVO_AABC/derived).")
    return AABC_EXAMPLE

# one flat import namespace across the package, as the scripts were written against
for _d in ("lib", "detect", "analysis", "analysis/depth", "tractography", "validation", "invivo",
           "figures"):
    _p = os.path.join(HERE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
