#!/usr/bin/env python
"""
Patch-based training script for 2× SRDRM on UFO-120 dataset.
Author: Mayank Meghwal
"""

import os
import datetime
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils.plot_utils import save_val_samples
from utils.loss_utils import total_gen_loss
from utils.data_utils import deprocess, preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------------------------------------
# PATCH SETTINGS
# -----------------------------------------------------------

LR_PATCH = 128           # LR patch 128×128
HR_PATCH = LR_PATCH * 2  # HR patch 256×256

# -----------------------------------------------------------
# DATASET PATH
# -----------------------------------------------------------

DATA_PATH = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Code\Data\UFO-120"

train_lr = os.path.join(DATA_PATH, "train_val", "lr_2x")
train_hr = os.path.join(DATA_PATH, "train_val", "hr")

lr_files = sorted(os.listdir(train_lr))
hr_files = sorted(os.listdir(train_hr))

num_samples = min(len(lr_files), len(hr_files))
print(f"Loaded {num_samples} LR-HR image pairs")

# -----------------------------------------------------------
# UTILITY: LOAD FULL IMAGES
# -----------------------------------------------------------

def load_image(path):
    img = load_img(path)
    return img_to_array(img).astype(np.float32)

# -----------------------------------------------------------
# PATCH EXTRACTION FUNCTION
# -----------------------------------------------------------

def extract_patch_pair(img_lr, img_hr):
    """Extracts aligned LR-HR patches"""
    h_lr, w_lr, _ = img_lr.shape
    h_hr, w_hr, _ = img_hr.shape

    # Random LR patch top-left
    x_lr = np.random.randint(0, w_lr - LR_PATCH)
    y_lr = np.random.randint(0, h_lr - LR_PATCH)

    # Corresponding HR patch
    x_hr = x_lr * 2
    y_hr = y_lr * 2

    lr_patch = img_lr[y_lr:y_lr+LR_PATCH, x_lr:x_lr+LR_PATCH]
    hr_patch = img_hr[y_hr:y_hr+HR_PATCH, x_hr:x_hr+HR_PATCH]

    return lr_patch, hr_patch

# -----------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------

from nets.gen_models import SRDRM_gen

lr_shape = (LR_PATCH, LR_PATCH, 3)
hr_shape = (HR_PATCH, HR_PATCH, 3)

model = SRDRM_gen(lr_shape, hr_shape, SCALE=2).create_model()

optimizer = Adam(0.0002, 0.5)
model.compile(optimizer=optimizer, loss=total_gen_loss)

# -----------------------------------------------------------
# OUTPUT FOLDERS
# -----------------------------------------------------------

dataset_name = "UFO120_2x_patches"

checkpoint_dir = os.path.join(
    r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Code\SRDRM\checkpoints",
    dataset_name
)

samples_dir = os.path.join(
    r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Code\SRDRM\images",
    dataset_name
)

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

print("\n=== TRAINING CONFIGURATION ===")
print("Patch-based training enabled")
print("LR Patch:", LR_PATCH)
print("HR Patch:", HR_PATCH)
print("Dataset Path:", DATA_PATH)
print("Checkpoints:", checkpoint_dir)
print("==============================\n")

# -----------------------------------------------------------
# TRAINING PARAMETERS
# -----------------------------------------------------------

num_epochs = 40
batch_size = 4           # CAN USE HIGHER BATCH SIZE NOW
steps_per_epoch = 500    # steps per epoch (not total images)

# -----------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------

for epoch in range(1, num_epochs + 1):

    for step in range(steps_per_epoch):

        lr_batch = []
        hr_batch = []

        # Create batch of LR/HR patches
        for _ in range(batch_size):

            idx = np.random.randint(0, num_samples)

            img_lr = load_image(os.path.join(train_lr, lr_files[idx]))
            img_hr = load_image(os.path.join(train_hr, hr_files[idx]))

            lr_patch, hr_patch = extract_patch_pair(img_lr, img_hr)

            lr_batch.append(lr_patch)
            hr_batch.append(hr_patch)

        lr_batch = preprocess(np.array(lr_batch))
        hr_batch = preprocess(np.array(hr_batch))

        loss = model.train_on_batch(lr_batch, hr_batch)

        if step % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Step {step}/{steps_per_epoch} | Loss: {loss:.4f}")

    # Save model checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"model_{epoch}")
    with open(ckpt_path + "_.json", "w") as jf:
        jf.write(model.to_json())
    model.save_weights(ckpt_path + "_.h5")

    print(f"Saved checkpoint at epoch {epoch}")

print("\nTraining Completed Successfully!\n")
