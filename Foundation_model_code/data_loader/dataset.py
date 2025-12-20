from __future__ import annotations
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import monai
import h5py
from typing import Dict, Any


class Lymphoma_Dataset_slide_lvl(monai.data.PersistentDataset):
    """
    Slide-level dataset that reads per-slide HDF5 feature files.

    Returns items:
        {
            "patient_id": patient_id or -1,
            "slide_id":   slide_id,
            "image":      np.ndarray [patches, feat_dim],
            "coords":     -1,
            "label":      int or -1,  (torch.tensor(int) if present)
            "tumor_ratio":       -1,
            "tumor_patch_count": -1,
        }
    """

    def __init__(self, params: Dict[str, Any], data_frame=None, device=None, shuffle: bool = True):
        if data_frame is None:
            raise ValueError("data_frame must be provided")

        # Set attributes (kept for compatibility with your code)
        self.__dict__.update(locals())

        self.feature_dir = Path(params["files"]["data_location"])
        self.label_name = params["inputs"]["label_name"]
        self.data = data_frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        slide_row = self.data.loc[idx]
        slide_id = slide_row["slide_id"]

        patient_id = slide_row.get("patient_id", -1)

        # Single-label expected (multi-label would be tensor of shape [K])
        label = slide_row.get(self.label_name, -1)
        try:
            label = torch.tensor(int(label))
        except Exception:
            label = torch.tensor(-1)

        # Try opening .hdf5 first, then .h5
        full_path_hdf5 = self.feature_dir / f"{slide_id}.hdf5"
        full_path_h5 = self.feature_dir / f"{slide_id}.h5"

        features_filter = None
        for p in (full_path_hdf5, full_path_h5):
            if p.exists():
                try:
                    with h5py.File(p, "r") as file:
                        feats = file["features"][:]  # [patches, feat_dim]
                        mask = ~np.isnan(feats).any(axis=1)
                        features_filter = feats[mask]
                    break
                except (OSError, FileNotFoundError):
                    continue

        if features_filter is None:
            raise FileNotFoundError(
                f"Neither {full_path_hdf5} nor {full_path_h5} could be opened or contained features."
            )

        # Optional shuffle
        if self.shuffle and features_filter.shape[0] > 1:
            index = np.random.permutation(features_filter.shape[0])
            features_filter = features_filter[index]

        return {
            "patient_id": patient_id,
            "slide_id": slide_id,
            "image": features_filter,  # numpy; convert to torch in collate if desired
            "coords": -1,
            "label": label,
            "tumor_ratio": -1,
            "tumor_patch_count": -1,
        }
    