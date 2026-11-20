"""Read the labelled mmWaveTryOn clips and construct the eight participant folds."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset


PARTICIPANTS = (
    1210175720, 1210181043, 1211155105, 1211170323,
    1212175544, 1212180933, 1216164252, 1216170002,
)
ACTION_IDS = (1, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18)
JOINTS = (0, 11, 13, 15, 12, 14, 16, 23, 24, 25, 26, 27, 28)
FRAMES = 20


def split(test_person_id):
    """Return the published train, validation and test participant IDs for one fold."""
    test_id = int(test_person_id)
    index = PARTICIPANTS.index(test_id)
    # For the last test participant the original split uses the preceding person.
    validation_id = PARTICIPANTS[index + 1] if index + 1 < len(PARTICIPANTS) else PARTICIPANTS[-2]
    train_ids = tuple(p for p in PARTICIPANTS if p not in (test_id, validation_id))
    return train_ids, (validation_id,), (test_id,)


class Dataset(TorchDataset):
    """Load contiguous 20-frame clips from a Data_Processing directory.

    A sample has an integer label and a clip_id (its first image filename).
    Radar and both fusion models include images [time, channel, height, width].
    The concat model uses the original RGB maps scaled to [0, 1]. The radar
    and attention models use resized and normalized RGB maps.
    The skeleton-only benchmarks use the final pose [1, 13, 2] of each clip.
    Fusion uses every pose [20, 13, 2]. The x and y coordinates come directly
    from the normalized CSV pose columns.
    """

    def __init__(self, data_dir, person_ids, modality, image_size=None):
        if modality not in ("skeleton", "radar", "concat", "fusion"):
            raise ValueError("modality must be skeleton, radar, concat or fusion (attention)")
        self.modality = modality
        self.image_size = image_size
        if self.image_size is None:
            self.image_size = {"radar": 64, "fusion": 224}.get(modality)
        if self.image_size is not None and (not isinstance(self.image_size, int) or self.image_size < 1):
            raise ValueError("image_size must be a positive integer")

        root = Path(data_dir)
        self.image_directory = root / "All_RD_imgs"
        pose_columns = [f"pose_{joint}_{axis}" for joint in JOINTS for axis in ("x", "y")]
        columns = ["person_id", "action_id", "frame", "imagename"]
        if modality != "radar":
            columns += pose_columns
        labels = pd.read_csv(root / "labels.csv", usecols=columns)
        people = {int(p) for p in person_ids}
        if not people:
            raise ValueError("person_ids cannot be empty")
        missing = people - set(labels["person_id"])
        if missing:
            raise ValueError(f"Participants missing from labels.csv: {sorted(missing)}")
        selected = labels[labels["person_id"].isin(people)]
        if selected.isna().any().any():
            raise ValueError("Missing values in the selected frame labels or pose coordinates")
        if selected["imagename"].duplicated().any():
            raise ValueError("A frame filename occurs more than once in the selected labels")
        unknown = set(selected["action_id"]) - set(ACTION_IDS)
        if unknown:
            raise ValueError(f"Unrecognized mmWaveTryOn action IDs: {sorted(unknown)}")

        self.clips = []
        # Preserve the CSV frame order within each participant/action, as in training.
        for person_id, participant in selected.groupby("person_id", sort=False):
            for action_id, group in participant.groupby("action_id", sort=False):
                if len(group) % FRAMES:
                    raise ValueError(f"Incomplete {FRAMES}-frame clip for {person_id}, action {action_id}")
                frame_numbers = group["frame"].to_numpy().reshape(-1, FRAMES)
                if not np.all(np.diff(frame_numbers, axis=1) == 1):
                    raise ValueError(f"Nonconsecutive frames for {person_id}, action {action_id}")
                names = group["imagename"].to_numpy().reshape(-1, FRAMES)
                poses = None
                if modality != "radar":
                    xy = group[pose_columns].to_numpy(dtype=np.float32)
                    if not np.isfinite(xy).all():
                        raise ValueError(f"Nonfinite pose coordinates for {person_id}, action {action_id}")
                    poses = torch.from_numpy(xy.reshape(-1, FRAMES, len(JOINTS), 2))
                    if modality == "skeleton":
                        poses = poses[:, -1:]
                label = ACTION_IDS.index(action_id)
                for index, filenames in enumerate(names):
                    pose = None if poses is None else poses[index]
                    self.clips.append((filenames, pose, label))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        names, pose, label = self.clips[index]
        sample = {"label": label, "clip_id": names[0]}
        if pose is not None:
            sample["skeletons"] = pose
        if self.modality != "skeleton":
            frames = []
            for name in names:
                with Image.open(self.image_directory / name) as source:
                    image = source.convert("RGB")
                    if self.image_size is not None:
                        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                    frames.append(np.asarray(image))
            images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255)
            if self.modality != "concat":
                mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
                std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
                images.sub_(mean).div_(std)
            sample["images"] = images
        return sample
