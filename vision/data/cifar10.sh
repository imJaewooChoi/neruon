#!/bin/bash

CIFAR10_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

DATA_DIR="classification"

mkdir -p $DATA_DIR
echo "Downloading CIFAR-10 Dataset..."
wget $CIFAR10_URL -O $DATA_DIR/cifar-10-python.tar.gz

echo "Extracting CIFAR-10 Dataset..."
tar -xzvf $DATA_DIR/cifar-10-python.tar.gz -C $DATA_DIR

cd $DATA_DIR
python load_cifar10.py $DATA_DIR/cifar-10-batches-py $DATA_DIR

echo "CIFAR-10 Dataset Saved."

