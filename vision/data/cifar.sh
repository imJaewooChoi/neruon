#!/bin/bash

echo "Enter the path where you want to save CIFAR-100 data:"
read data_path

echo "Enter the path where you want to save the images:"
read save_path

if [ ! -d "$data_path" ]; then
    mkdir -p "$data_path"
fi

if [ ! -d "$save_path" ]; then
    mkdir -p "$save_path"
fi

wget -P "$data_path" https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf "$data_path/cifar-100-python.tar.gz" -C "$data_path"

pip install numpy opencv-python

python save_cifar_images.py "$data_path/cifar-100-python" "$save_path"

echo "CIFAR-100 processing completed."
