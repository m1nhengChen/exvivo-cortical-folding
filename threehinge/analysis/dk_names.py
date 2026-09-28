"""Desikan-Killiany raw FreeSurfer label -> publication region name."""
# --- project paths and import roots: see src/threehinge/_env.py ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import _env

DK_DISPLAY = {
    "bankssts": "Banks of superior temporal sulcus",
    "caudalanteriorcingulate": "Caudal anterior cingulate",
    "caudalmiddlefrontal": "Caudal middle frontal",
    "cuneus": "Cuneus",
    "entorhinal": "Entorhinal",
    "fusiform": "Fusiform",
    "inferiorparietal": "Inferior parietal",
    "inferiortemporal": "Inferior temporal",
    "isthmuscingulate": "Isthmus cingulate",
    "lateraloccipital": "Lateral occipital",
    "lateralorbitofrontal": "Lateral orbitofrontal",
    "lingual": "Lingual",
    "medialorbitofrontal": "Medial orbitofrontal",
    "middletemporal": "Middle temporal",
    "parahippocampal": "Parahippocampal",
    "paracentral": "Paracentral",
    "parsopercularis": "Pars opercularis",
    "parsorbitalis": "Pars orbitalis",
    "parstriangularis": "Pars triangularis",
    "pericalcarine": "Pericalcarine",
    "postcentral": "Postcentral",
    "posteriorcingulate": "Posterior cingulate",
    "precentral": "Precentral",
    "precuneus": "Precuneus",
    "rostralanteriorcingulate": "Rostral anterior cingulate",
    "rostralmiddlefrontal": "Rostral middle frontal",
    "superiorfrontal": "Superior frontal",
    "superiorparietal": "Superior parietal",
    "superiortemporal": "Superior temporal",
    "supramarginal": "Supramarginal",
    "frontalpole": "Frontal pole",
    "temporalpole": "Temporal pole",
    "transversetemporal": "Transverse temporal",
    "insula": "Insula",
}
