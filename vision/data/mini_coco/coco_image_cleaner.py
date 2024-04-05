import json
import os


class COCOImageCleaner:
    def __init__(self, annotation_file, image_dir):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.annotated_images = set()

    def load_annotated_images(self):
        with open(self.annotation_file) as f:
            data = json.load(f)
            self.annotated_images = set([img["file_name"] for img in data["images"]])
        print(
            f"Loaded {len(self.annotated_images)} annotated images from {self.annotation_file}"
        )

    def clean_images(self):
        self.load_annotated_images()
        for img_file in os.listdir(self.image_dir):
            if img_file not in self.annotated_images:
                os.remove(os.path.join(self.image_dir, img_file))
                print(f"Deleted {img_file}")
        print("Image cleaning complete.")

    def verify_images(self):
        unannotated_files = []
        for img_file in os.listdir(self.image_dir):
            if img_file not in self.annotated_images:
                unannotated_files.append(img_file)
        if not unannotated_files:
            print("Verification complete: All images are annotated.")
        else:
            print(
                f"Verification failed: Found {len(unannotated_files)} unannotated images."
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python coco_image_cleaner.py annotation_file image_dir")
        sys.exit(1)

    annotation_file = sys.argv[1]
    image_dir = sys.argv[2]

    cleaner = COCOImageCleaner(annotation_file, image_dir)
    cleaner.clean_images()
    cleaner.verify_images()
