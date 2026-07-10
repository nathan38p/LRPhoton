from pathlib import Path

import h5py
import numpy as np


Q_MAP_DATASET_CANDIDATES = (
    "/entry_0000/instrument/detector/q_nm_inverse",
    "/entry_0000/instrument/detector/q",
    "/q_nm_inverse",
    "/q_nm",
)


def is_real_geometry_h5(filename):
    path = Path(filename)
    if not path.exists() or path.suffix.lower() not in {".h5", ".hdf5"}:
        return False
    if "real_geometry" in path.stem.lower():
        return True

    try:
        with h5py.File(path, "r") as h5:
            for obj in (h5, h5.get("/entry_0000/instrument/detector/data")):
                if obj is None:
                    continue
                attrs = obj.attrs
                if bool(attrs.get("rectified_regular_detector_grid", False)):
                    return True
                panel = attrs.get("panel", "")
                if isinstance(panel, bytes):
                    panel = panel.decode(errors="ignore")
                if str(panel).lower() == "real":
                    return True
    except Exception:
        return False

    return False


def read_h5_q_map(filename, expected_shape=None):
    path = Path(filename)
    if not path.exists() or path.suffix.lower() not in {".h5", ".hdf5"}:
        return None

    with h5py.File(path, "r") as h5:
        for name in Q_MAP_DATASET_CANDIDATES:
            if name in h5:
                q_map = np.asarray(h5[name][...], dtype=float)
                if q_map_matches_shape(q_map, expected_shape):
                    return q_map

        found = []

        def collect(name, obj):
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 2:
                return
            lower = name.lower()
            if "q" not in lower:
                return
            q_map = np.asarray(obj[...], dtype=float)
            if q_map_matches_shape(q_map, expected_shape):
                found.append(q_map)

        h5.visititems(collect)
        return found[0] if found else None


def q_map_matches_shape(q_map, expected_shape):
    if q_map.ndim != 2:
        return False
    if expected_shape is None:
        return True
    return tuple(q_map.shape) == tuple(expected_shape)
