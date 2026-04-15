import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(train: bool = True, img_size: int = 224) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), # ImageNet stats
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def sample_frames(video_path: str, num_frames: int) -> List:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()

    # Ensure num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else None)
    return frames[:num_frames]


class ViolenceDataset(Dataset):
    LABEL_MAP = {"Violence": 1, "NonViolence": 0}

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_ratio: float = 0.2,
        num_frames: int = 16,
        img_size: int = 224,
        seed: int = 42,
    ):
        self.num_frames = num_frames
        self.transform = build_transforms(train=(split == "train"), img_size=img_size)

        samples: List[Tuple[str, int]] = []
        for cls_name, label in self.LABEL_MAP.items():
            cls_dir = Path(root) / cls_name
            for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
                for p in cls_dir.glob(ext):
                    samples.append((str(p), label))

        random.seed(seed)
        random.shuffle(samples)
        n_val = int(len(samples) * val_ratio)

        if split == "val":
            self.samples = samples[:n_val]
        else:
            self.samples = samples[n_val:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        frames = sample_frames(path, self.num_frames)

        if not frames or frames[0] is None:
            clip = torch.zeros(self.num_frames, 3, 224, 224)
        else:
            clip = torch.stack([self.transform(f) for f in frames])  # (T, C, H, W)

        return clip, label