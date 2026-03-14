#!/usr/bin/env python
import os
import time
import ntpath
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

from utils.data_utils import preprocess, deprocess


# 1. Correct UFO-120 TEST LR directory

lr_test_dir = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Code\Data\UFO-120\TEST\lrd"
hr_test_dir = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Code\Data\UFO-120\TEST\hr"

lr_images = sorted(os.listdir(lr_test_dir))
print(f"{len(lr_images)} LR test images found")


# 2. Load SRDRM model (your trained checkpoint)

model_name = "srdrm"
ckpt_name = "model_40_"

checkpoint_dir = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Code\SRDRM\checkpoints\UFO120_2x_patches"

model_json = os.path.join(checkpoint_dir, ckpt_name + ".json")
model_h5   = os.path.join(checkpoint_dir, ckpt_name + ".h5")

with open(model_json, "r") as f:
    generator = model_from_json(f.read())

generator.load_weights(model_h5)
print("Loaded SRDRM model successfully")


# 3. Output folder for new SRDRM results

output_dir = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Results\SRDRM\2x"
os.makedirs(output_dir, exist_ok=True)


# 4. Generate SUPER-RES results with matched names

times = []

for filename in lr_images:

    base_name = filename.split(".")[0]        # e.g., set_f0
    lr_path = os.path.join(lr_test_dir, filename)

    # load LR image
    img_lr = cv2.imread(lr_path)
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)

    # model expects 128×128 input patches
    img_lr_resized = cv2.resize(img_lr, (128, 128))

    im = preprocess(img_lr_resized)
    im = np.expand_dims(im, axis=0)

    t0 = time.time()
    gen = generator.predict(im)[0]
    gen = deprocess(gen)
    times.append(time.time() - t0)

    # convert to displayable image
    gen_img = (gen * 255).astype(np.uint8)
    gen_img = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)

    # upscale final SR result to 640×480
    gen_img = cv2.resize(gen_img, (640, 480))

    # SAVE WITH EXACT MATCHING NAME
    save_path = os.path.join(output_dir, base_name + "_gen.jpg")
    cv2.imwrite(save_path, gen_img)

    print("Generated:", base_name)


# 5. Summary

print(f"\nTotal generated SR images: {len(lr_images)}")
print("Saved in:", output_dir)
print(f"Average inference time: {np.mean(times):.4f} sec\n")
