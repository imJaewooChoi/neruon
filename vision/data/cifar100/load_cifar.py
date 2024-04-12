import pickle
import numpy as np
import cv2
import os


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def save_images_from_batch(batch, label_dir):
    for i, img_array in enumerate(batch[b"data"]):
        img_label_index = batch[b"fine_labels"][i]
        img_class_dir = os.path.join(label_dir, str(img_label_index))
        if not os.path.exists(img_class_dir):
            os.makedirs(img_class_dir)

        img_array = img_array.reshape(3, 32, 32)
        img_array = img_array.transpose(1, 2, 0)
        img_path = os.path.join(img_class_dir, f"image_{i}.png")
        cv2.imwrite(img_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))


def save_cifar_images(cifar100_dir, save_dir):
    train_file = os.path.join(cifar100_dir, "train")
    test_file = os.path.join(cifar100_dir, "test")

    train_save_dir = os.path.join(save_dir, "train")
    train_data = unpickle(train_file)
    save_images_from_batch(train_data, train_save_dir)

    test_save_dir = os.path.join(save_dir, "test")
    test_data = unpickle(test_file)
    save_images_from_batch(test_data, test_save_dir)

    print(
        "CIFAR-100 images have been saved by numeric labels in the specified directory."
    )


cifar100_dir = "/workspace/neuron/vision/data/cifar100/cifar-100-python"
save_dir = "/workspace/neuron/vision/data/cifar100"
save_cifar_images(cifar100_dir, save_dir)
