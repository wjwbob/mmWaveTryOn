import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from dataset import Dataset, JOINTS, PARTICIPANTS, split


class BenchmarkInputsTest(unittest.TestCase):
    def test_held_out_participants(self):
        for person in PARTICIPANTS:
            train, validation, test = split(person)
            self.assertEqual((len(train), len(validation), test), (6, 1, (person,)))
            self.assertEqual(set(train + validation + test), set(PARTICIPANTS))
        self.assertEqual(split(PARTICIPANTS[-1])[1], (PARTICIPANTS[-2],))

    def test_model_specific_clip_transforms(self):
        person = PARTICIPANTS[0]
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            images = root / "All_RD_imgs"
            images.mkdir()
            rows = []
            for frame in range(20):
                name = f"{frame}.png"
                Image.new("RGB", (12, 8), (51, 102, 153)).save(images / name)
                row = dict(person_id=person, action_id=1, frame=frame, imagename=name)
                for joint in JOINTS:
                    row[f"pose_{joint}_x"] = frame / 20
                    row[f"pose_{joint}_y"] = joint / 33
                rows.append(row)
            with (root / "labels.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            skeleton = Dataset(root, [person], "skeleton")[0]
            concat = Dataset(root, [person], "concat")[0]
            attention = Dataset(root, [person], "fusion")[0]
            radar = Dataset(root, [person], "radar")[0]
            self.assertEqual(skeleton["skeletons"].shape, (1, 13, 2))
            self.assertEqual(concat["skeletons"].shape, (20, 13, 2))
            torch.testing.assert_close(skeleton["skeletons"], concat["skeletons"][-1:])
            torch.testing.assert_close(attention["skeletons"], concat["skeletons"])
            rgb = torch.tensor([0.2, 0.4, 0.6])
            normalized = (rgb - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
            for sample, size, target in (
                (concat, (8, 12), rgb), (attention, (224, 224), normalized), (radar, (64, 64), normalized)
            ):
                self.assertEqual(sample["images"].shape, (20, 3, *size))
                torch.testing.assert_close(sample["images"][0, :, 0, 0], target)


if __name__ == "__main__":
    unittest.main()
