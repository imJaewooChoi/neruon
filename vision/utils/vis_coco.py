import os
import cv2
import numpy as np
from pycocotools.coco import COCO


class COCOVisualizer:
    def __init__(self, annotation_path: str, image_dir: str):
        self.coco = COCO(annotation_path)
        self.image_dir = image_dir
        self.categories = {
            cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }
        self.colors = {
            cat_id: [np.random.randint(0, 256) for _ in range(3)]
            for cat_id in self.categories.keys()
        }

    def visualize_image(self, image_id: int, output_dir: str):
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = cv2.imread(image_path)
        assert image is not None, f"Image not found at '{image_path}'."

        overlay = image.copy()
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            color = self.colors[ann["category_id"]]
            if "segmentation" in ann and type(ann["segmentation"]) == list:
                for seg in ann["segmentation"]:
                    poly = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [poly], color=color)

        alpha = 0.5
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        for ann in anns:
            if "bbox" in ann:
                bbox = ann["bbox"]
                label = self.categories[ann["category_id"]]
                cv2.rectangle(
                    image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                    (0, 0, 255),
                    2,
                )
                label_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1
                )[0]
                label_bg = (
                    (int(bbox[0]), int(bbox[1] - label_size[1] - 4)),
                    (int(bbox[0] + label_size[0]), int(bbox[1])),
                )
                cv2.rectangle(image, label_bg[0], label_bg[1], (0, 0, 0), -1)
                cv2.putText(
                    image,
                    label,
                    (int(bbox[0]), int(bbox[1] - 4)),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (255, 255, 255),
                    1,
                )

        output_path = os.path.join(output_dir, image_info["file_name"])
        cv2.imwrite(output_path, image)
        print(f"Image with ID {image_id} visualized and saved to {output_path}")


if __name__ == "__main__":
    annotation_path = "/workspace/dl_choi/vision/data/mini_coco/train.json"
    image_dir = "/workspace/dl_choi/vision/data/mini_coco/val2017"
    output_dir = "/workspace/dl_choi/vision/utils/assets"
    image_id = 402720
    visualizer = COCOVisualizer(annotation_path, image_dir)
    visualizer.visualize_image(image_id, output_dir)
