import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        sample_weight: float = 1.0,
    ):
        """
        Args:
            path: Path to folder containing .mat files
            split: "train", "val", or "test"
            sample_weight: Sampling weight for this dataset
            tokenizer_version: CLIP tokenizer version
        """
        assert split in ("train", "val", "test")
        
        self.path = path
        self.split = split
        self.sample_weight = sample_weight
        
        # Load .mat files from path/<split>/
        mat_files = sorted(glob.glob(os.path.join(self.path, self.split, "*.mat")))
        if len(mat_files) == 0:
            raise ValueError(f"No .mat files found in {os.path.join(self.path, self.split)}")
        
        self.file_list = mat_files
        print(f"BrainMRIDataset ({split}): loaded {len(self.file_list)} samples from {os.path.join(self.path, self.split)}")

    def __len__(self) -> int:
        return int(len(self.file_list) * self.sample_weight)

    def __getitem__(self, idx: int):
        if self.sample_weight < 1.0:
            real_idx = int(idx / self.sample_weight) % len(self.file_list)
        else:
            real_idx = idx % len(self.file_list)
        
        file_path = self.file_list[real_idx]

        mat_data = loadmat(file_path)
        image = mat_data["image"]
        label = mat_data["label"]
        instruction = mat_data["instruction"]

        instruction = str(instruction.item()).strip()

        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(torch.float32)
        label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).to(torch.float32)

        image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
        label = F.interpolate(label, size=(256, 256), mode="bilinear", align_corners=False)

        image = 2 * (image / 255.0) - 1
        label = 2 * (label / 255.0) - 1

        image = image.repeat(1, 3, 1, 1).squeeze(0)  # [3,256,256]
        label = label.repeat(1, 3, 1, 1).squeeze(0)  # [3,256,256]

        return dict(edited=label, edit=dict(c_concat=image, c_crossattn=instruction))
