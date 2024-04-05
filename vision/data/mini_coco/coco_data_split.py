import json
import numpy as np


class COCODatasetSplitter:
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        self.data = None
        self.load_annotations()

    def load_annotations(self):
        with open(self.annotation_file, "r") as f:
            self.data = json.load(f)

    def split_dataset(self, train_ratio=0.7, valid_ratio=0.2):
        assert (
            train_ratio + valid_ratio < 1
        ), "Train and validation ratios must sum to less than 1."

        image_ids = [img["id"] for img in self.data["images"]]
        np.random.shuffle(image_ids)

        train_idx = int(len(image_ids) * train_ratio)
        val_idx = train_idx + int(len(image_ids) * valid_ratio)

        train_ids = image_ids[:train_idx]
        val_ids = image_ids[train_idx:val_idx]
        test_ids = image_ids[val_idx:]

        self.save_split_data(train_ids, "train.json")
        self.save_split_data(val_ids, "valid.json")
        self.save_split_data(test_ids, "test.json")

    def filter_data(self, ids):
        images = [img for img in self.data["images"] if img["id"] in ids]
        annotations = [
            ann for ann in self.data["annotations"] if ann["image_id"] in ids
        ]
        return {
            "images": images,
            "annotations": annotations,
            "categories": self.data["categories"],
        }

    def save_split_data(self, ids, filename):
        split_data = self.filter_data(ids)
        with open(filename, "w") as f:
            json.dump(split_data, f, indent=4)
        print(f"{filename} has been saved with {len(ids)} images.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print(
            "Usage: python coco_dataset_splitter.py annotation_file train_ratio valid_ratio test_ratio"
        )
        sys.exit(1)

    annotation_file = sys.argv[1]
    train_ratio = float(sys.argv[2])
    valid_ratio = float(sys.argv[3])
    test_ratio = float(sys.argv[4])

    assert (
        round(train_ratio + valid_ratio + test_ratio) == 1.0
    ), "Train, validation, and test ratios must sum to 1."

    splitter = COCODatasetSplitter(annotation_file)
    splitter.split_dataset(train_ratio=train_ratio, valid_ratio=valid_ratio)
