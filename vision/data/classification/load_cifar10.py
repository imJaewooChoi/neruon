import cv2
import numpy as np
import os
import pickle


def load_cifar10_batch(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    data = dict[b"data"]
    labels = np.array(dict[b"labels"])
    return data, labels


def save_images(data, labels, directory):
    for i, (img, label) in enumerate(zip(data, labels)):
        img = img.reshape(3, 32, 32).transpose(1, 2, 0)
        label_dir = os.path.join(directory, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        cv2.imwrite(os.path.join(label_dir, f"{i}.png"), img)


def load_and_save_cifar10(data_dir, save_dir):
    train_dir = os.path.join(save_dir, "train")
    for i in range(1, 6):
        data_batch, label_batch = load_cifar10_batch(
            os.path.join(data_dir, f"data_batch_{i}")
        )
        save_images(data_batch, label_batch, train_dir)

    test_dir = os.path.join(save_dir, "test")
    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, "test_batch"))
    save_images(test_data, test_labels, test_dir)


cifar10_dir = "/workspace/ai_study/vision/data/classification/cifar-10-batches-py"

save_dir = "/workspace/ai_study/vision/data/classification/cifar10"

load_and_save_cifar10(cifar10_dir, save_dir)
