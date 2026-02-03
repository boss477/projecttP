"""
Data preprocessing and loading for Cholec80 and EndoVis datasets.
"""

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ===========================
# Cholec80 PHASE LABEL MAPPING
# ===========================
PHASE_TO_ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}

ID_TO_PHASE = {v: k for k, v in PHASE_TO_ID.items()}


# ✅ Added: robust alias mapping (ONLY addition)
PHASE_ALIASES = {
    "Preparation": "Preparation",
    "Calot Triangle Dissection": "CalotTriangleDissection",
    "Clipping and Cutting": "ClippingCutting",
    "Gallbladder Dissection": "GallbladderDissection",
    "Gallbladder Packaging": "GallbladderPackaging",
    "Cleaning and Coagulation": "CleaningCoagulation",
    "Gallbladder Retraction": "GallbladderRetraction",
}


# ===========================
# Cholec80 DATASET
# ===========================
class Cholec80Dataset(Dataset):
    """
    PyTorch Dataset for Cholec80 surgical phase recognition.
    """

    def __init__(
        self,
        root_dir: str,
        video_ids: List[str],
        sequence_length: int = 5,
        frame_skip: int = 1,
        transform=None,
        mode: str = "train",
    ):
        self.root_dir = Path(root_dir)
        self.video_ids = video_ids
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.transform = transform
        self.mode = mode

        self.samples = self._load_annotations()

        if len(self.samples) == 0:
            raise RuntimeError(
                "❌ No training samples found.\n"
                "Your phase annotations are present but were not parsed correctly.\n"
                "Make sure phase names match the official Cholec80 labels."
            )

        print(f"Loaded {len(self.samples)} sequences from {len(video_ids)} videos")

    def _load_annotations(self) -> List[Dict]:
        samples = []

        for video_id in self.video_ids:
            ann_file = self.root_dir / "annotations" / f"{video_id}-phase.txt"
            video_path = self.root_dir / "videos" / f"{video_id}.mp4"

            if not ann_file.exists():
                print(f"[WARN] Missing annotation file: {ann_file}")
                continue

            if not video_path.exists():
                print(f"[WARN] Missing video file: {video_path}")
                continue

            frame_labels = []

            with open(ann_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()

                    # Skip header like: "Frame Phase"
                    if not parts[0].isdigit():
                        continue

                    frame_num = int(parts[0]) - 1  # ✅ FIX: OpenCV is 0-indexed
                    raw_phase = " ".join(parts[1:]).strip()  # ✅ FIX: multi-word phases

                    if raw_phase not in PHASE_ALIASES:
                        continue

                    phase_name = PHASE_ALIASES[raw_phase]
                    label = PHASE_TO_ID[phase_name]

                    frame_labels.append((frame_num, label))

            if len(frame_labels) < self.sequence_length:
                print(f"[WARN] Too few valid labels in {ann_file}")
                continue

            max_start = len(frame_labels) - self.sequence_length * self.frame_skip
            for i in range(0, max_start, self.sequence_length * self.frame_skip):
                frames = []
                labels = []

                for j in range(self.sequence_length):
                    idx = i + j * self.frame_skip
                    frames.append(frame_labels[idx][0])
                    labels.append(frame_labels[idx][1])

                samples.append(
                    {
                        "video_path": str(video_path),
                        "video_id": video_id,
                        "frames": frames,
                        "labels": labels,
                    }
                )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        cap = cv2.VideoCapture(sample["video_path"])
        frames = []

        for frame_num in sample["frames"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_num}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))

            if self.transform:
                frame = self.transform(image=frame)["image"]
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

            frames.append(frame)

        cap.release()

        frames = torch.stack(frames)
        labels = torch.tensor(sample["labels"], dtype=torch.long)

        return frames, labels


# ===========================
# ENDOVIS DATASET
# ===========================
class EndoVisDataset(Dataset):
    """
    PyTorch Dataset for EndoVis tool detection (YOLO format).
    """

    def __init__(self, root_dir: str, transform=None, mode: str = "train"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode

        self.image_dir = self.root_dir / "images"
        self.ann_dir = self.root_dir / "annotations"
        self.image_files = sorted(self.image_dir.glob("*.jpg"))

        print(f"Loaded {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        ann_path = self.ann_dir / f"{img_path.stem}.txt"
        bboxes = []

        if ann_path.exists():
            with open(ann_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    try:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                    except ValueError:
                        continue

                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    bboxes.append([class_id, x1, y1, x2, y2])

        bboxes = np.array(bboxes) if bboxes else np.zeros((0, 5))

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes[:, 1:] if len(bboxes) else [],
                class_labels=bboxes[:, 0] if len(bboxes) else [],
            )
            image = transformed["image"]

            if transformed.get("bboxes"):
                bboxes = np.array(
                    [[lbl] + list(box) for box, lbl in zip(
                        transformed["bboxes"], transformed["class_labels"]
                    )]
                )
            else:
                bboxes = np.zeros((0, 5))
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, bboxes


# ===========================
# TRANSFORMS
# ===========================
def get_cholec80_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_endovis_transforms(mode="train"):
    return A.Compose(
        [
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5 if mode == "train" else 0.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )


# ===========================
# DATALOADERS
# ===========================
def create_data_loaders(
    data_dir: str,
    dataset_type: str = "cholec80",
    batch_size: int = 8,
    num_workers: int = 4,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader]:

    if dataset_type == "cholec80":
        video_dir = Path(data_dir) / "videos"
        video_ids = sorted(v.stem for v in video_dir.glob("*.mp4"))  # ✅ FIX

        split = int(len(video_ids) * train_split)
        train_ids, val_ids = video_ids[:split], video_ids[split:]

        train_dataset = Cholec80Dataset(
            data_dir,
            train_ids,
            transform=get_cholec80_transforms("train"),
            mode="train",
        )

        val_dataset = Cholec80Dataset(
            data_dir,
            val_ids,
            transform=get_cholec80_transforms("val"),
            mode="val",
        )

    else:
        full_dataset = EndoVisDataset(
            data_dir, transform=get_endovis_transforms("train")
        )
        split = int(len(full_dataset) * train_split)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [split, len(full_dataset) - split]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
