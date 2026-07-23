from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget


class MaskProject(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.mask_path = None
        layout = QVBoxLayout(self)
        image_box = QGroupBox("Image H5 / EDF")
        image_row = QHBoxLayout(image_box)
        self.image_edit = QLineEdit()
        image_button = QPushButton("Select")
        image_button.clicked.connect(self.select_image)
        image_row.addWidget(self.image_edit, 1)
        image_row.addWidget(image_button)
        mask_box = QGroupBox("Mask")
        mask_row = QHBoxLayout(mask_box)
        self.mask_edit = QLineEdit()
        mask_button = QPushButton("Select")
        mask_button.clicked.connect(self.select_mask)
        mask_row.addWidget(self.mask_edit, 1)
        mask_row.addWidget(mask_button)
        layout.addWidget(image_box)
        layout.addWidget(mask_box)
        self.apply_button = QPushButton("Apply mask and export H5")
        self.apply_button.clicked.connect(self.apply_mask)
        layout.addWidget(self.apply_button)
        self.status = QLabel("Select an image and a mask.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.h5 *.hdf5 *.edf)")
        if path:
            self.image_path = Path(path)
            self.image_edit.setText(str(self.image_path))

    def select_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select mask", "", "Masks (*.edf *.h5 *.hdf5)")
        if path:
            self.mask_path = Path(path)
            self.mask_edit.setText(str(self.mask_path))

    def read_array(self, path):
        if path.suffix.lower() == ".edf":
            import fabio
            return np.asarray(fabio.open(str(path)).data)
        import h5py
        with h5py.File(path, "r") as h5:
            candidates = []
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                    candidates.append((name, obj))
            h5.visititems(visit)
            if not candidates:
                raise ValueError("No 2D dataset found in H5 file.")
            return np.asarray(candidates[0][1][0] if candidates[0][1].ndim == 3 else candidates[0][1][()])

    def apply_mask(self):
        if self.image_path is None or self.mask_path is None:
            self.status.setText("Select an image and a mask first.")
            return
        try:
            import h5py
            image = np.asarray(self.read_array(self.image_path), dtype=float)
            mask = np.asarray(self.read_array(self.mask_path), dtype=bool)
            if image.shape != mask.shape:
                raise ValueError(f"Image shape {image.shape} differs from mask shape {mask.shape}.")
            output = image.copy()
            output[mask] = np.nan
            target, _ = QFileDialog.getSaveFileName(self, "Export masked H5", str(self.image_path.with_name(self.image_path.stem + "_masked.h5")), "HDF5 files (*.h5)")
            if not target:
                return
            with h5py.File(self.image_path, "r") as source, h5py.File(target, "w") as out:
                for key, value in source.attrs.items():
                    out.attrs[key] = value
                group = out.require_group("entry_0000/instrument/detector")
                dataset = group.create_dataset("data", data=output, compression="gzip")
                group.create_dataset("mask", data=mask.astype(np.uint8), compression="gzip")
                dataset.attrs["mask_applied"] = True
                dataset.attrs["mask_source"] = str(self.mask_path)
                out.attrs["mask_applied"] = True
                out.attrs["mask_source"] = str(self.mask_path)
            self.status.setText(f"Masked H5 exported: {Path(target).name}")
        except Exception as exc:
            self.status.setText(f"Mask export failed: {exc}")
