#!/usr/bin/env python
"""
Updated data utilities for SRDRM / UFO-120 dataset
All scipy.misc functions removed and replaced with cv2.

Maintainer (Modified): Mayank Meghwal
"""

from __future__ import division, absolute_import
import os
import random
import fnmatch
import numpy as np
import cv2


def deprocess(x):
    """Convert [-1, 1] → [0, 1]"""
    return (x + 1.0) * 0.5


def preprocess(x):
    """Convert [0, 255] → [-1, 1]"""
    return (x / 127.5) - 1.0


def augment(a_img, b_img):
    """Simple augmentation — random flips."""
    if random.random() < 0.25:
        a_img = np.fliplr(a_img)
        b_img = np.fliplr(b_img)
    if random.random() < 0.25:
        a_img = np.flipud(a_img)
        b_img = np.flipud(b_img)
    return a_img, b_img


def getPaths(data_dir):
    """Retrieve all image paths inside a folder."""
    exts = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.JPEG']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if fnmatch.fnmatch(filename, pattern):
                    image_paths.append(os.path.join(d, filename))
    return np.asarray(image_paths)


# FIXED: No more scipy.misc
def read_and_resize_pair(path_lr, path_hr, low_res=(60, 80), high_res=(480, 640)):
    """Load LR & HR image pair using cv2, resize correctly."""
    
    img_lr = cv2.imread(path_lr)
    img_hr = cv2.imread(path_hr)

    if img_lr is None:
        raise FileNotFoundError("LR image not found: " + path_lr)
    if img_hr is None:
        raise FileNotFoundError("HR image not found: " + path_hr)

    # Convert BGR → RGB
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)

    # Resize (cv2 uses W,H)
    img_lr = cv2.resize(img_lr, (low_res[1], low_res[0]))
    img_hr = cv2.resize(img_hr, (high_res[1], high_res[0]))

    return img_lr.astype(np.float32), img_hr.astype(np.float32)


class dataLoaderUSR():
    """Data loader compatible with USR-248 & UFO-120 structure."""

    def __init__(self, DATA_PATH, SCALE=4):
        self.SCALE = SCALE
        self.lr_res_, self.low_res_folder_ = self.get_lr_info()

        # Example folder structure:
        # train_val/
        #     lr_2x/
        #     hr/
        train_dir = val_dir = os.path.join(DATA_PATH, "train_val/")

        self.num_train, self.train_lr_paths, self.train_hr_paths = self.get_lr_hr_paths(train_dir)
        print(f"Loaded {self.num_train} training LR-HR pairs")

        self.num_val, self.val_lr_paths, self.val_hr_paths = self.get_lr_hr_paths(val_dir)

    def get_lr_info(self):
        """Return LR resolution and folder name based on scale."""
        if self.SCALE == 2:
            return (240, 320), "lr_2x/"
        elif self.SCALE == 8:
            return (60, 80), "lr_8x/"
        else:
            return (120, 160), "lr_4x/"  # Default 4x SR

    def get_lr_hr_paths(self, data_dir):
        """Load LR & HR image path lists."""
        lr_dir = os.path.join(data_dir, self.low_res_folder_)
        hr_dir = os.path.join(data_dir, "hr/")

        if not os.path.exists(lr_dir):
            raise FileNotFoundError("LR folder missing: " + lr_dir)
        if not os.path.exists(hr_dir):
            raise FileNotFoundError("HR folder missing: " + hr_dir)

        lr_path = sorted(os.listdir(lr_dir))
        hr_path = sorted(os.listdir(hr_dir))

        num_paths = min(len(lr_path), len(hr_path))
        lr_paths = [os.path.join(lr_dir, f) for f in lr_path[:num_paths]]
        hr_paths = [os.path.join(hr_dir, f) for f in hr_path[:num_paths]]

        return num_paths, lr_paths, hr_paths

    def load_batch(self, batch_size=1, data_augment=True):
        """Yield training batches."""
        self.n_batches = self.num_train // batch_size

        for i in range(self.n_batches - 1):
            batch_lr = self.train_lr_paths[i*batch_size:(i+1)*batch_size]
            batch_hr = self.train_hr_paths[i*batch_size:(i+1)*batch_size]

            imgs_lr, imgs_hr = [], []
            for idx in range(len(batch_lr)):
                img_lr, img_hr = read_and_resize_pair(batch_lr[idx], batch_hr[idx], low_res=self.lr_res_)

                if data_augment:
                    img_lr, img_hr = augment(img_lr, img_hr)

                imgs_lr.append(img_lr)
                imgs_hr.append(img_hr)

            imgs_lr = preprocess(np.array(imgs_lr))
            imgs_hr = preprocess(np.array(imgs_hr))

            yield imgs_lr, imgs_hr

    def load_val_data(self, batch_size=2):
        """Load random validation samples."""
        idx = np.random.choice(np.arange(self.num_val), batch_size, replace=False)

        imgs_lr, imgs_hr = [], []
        for i in idx:
            img_lr, img_hr = read_and_resize_pair(
                self.val_lr_paths[i],
                self.val_hr_paths[i],
                low_res=self.lr_res_
            )
            imgs_lr.append(img_lr)
            imgs_hr.append(img_hr)

        return preprocess(np.array(imgs_lr)), preprocess(np.array(imgs_hr))
