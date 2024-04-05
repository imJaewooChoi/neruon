mkdir -p mini_coco
cd mini_coco

echo "Downloading COCO validation images..."
wget -c http://images.cocodataset.org/zips/val2017.zip

echo "Downloading COCO annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Extracting downloaded files..."
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip

echo "Cleanup downloaded zip files..."
rm val2017.zip
rm annotations_trainval2017.zip

echo "COCO dataset is ready."

echo "Running image cleaner..."
python coco_image_cleaner.py annotations/instances_val2017.json val2017/

ANNOTATION_FILE=annotations/instances_val2017.json
TRAIN_RATIO=0.7
VALID_RATIO=0.2
TEST_RATIO=0.1

echo "Splitting COCO dataset..."
python coco_data_split.py $ANNOTATION_FILE $TRAIN_RATIO $VALID_RATIO $TEST_RATIO
echo "COCO dataset is split into train, valid, and test sets."