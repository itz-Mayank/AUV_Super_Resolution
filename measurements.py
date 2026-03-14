#!/usr/bin/env python
"""
Measure SSIM / PSNR / UIQM for ALL Enhancement + SR Models

Models supported:
 - IFM
 - UCM
 - Fusion
 - SRDRM 2×
 
Author: Mayank Meghwal
"""

import os
import numpy as np
from glob import glob
from ntpath import basename
from PIL import Image
import csv

from Code.PE.utils.uiqm_utils import getUIQM
from Code.PE.utils.ssm_psnr_utils import getSSIM, getPSNR

# CONFIG: SET YOUR DIRECTORIES HERE

GROUND_TRUTH = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Code\Data\UFO-120\TEST\hr"

MODEL_DIRS = {
    "IFM":    r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Results\IFM",
    "UCM":    r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Results\UCM",
    "GanEnhanced": r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Results\GanEnhanced",
    "SRDRM":  r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Results\SRDRM\2x"
}

OUTPUT_CSV = "metrics_results.csv"

# Image resize resolution for metric measurement
IM_W, IM_H = 256, 256 


# Helper Functions
def load_images(folder):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    return sorted(files)


def match_pairs(gt_list, model_list, suffix):
    """
    Match HR and Enhanced/SR images based on filename.
    Example:
       GT: set_f10.jpg
       IFM: set_f10.png  OR  set_f10_ifm.jpg (suffix matched)
       SRDRM: set_f10_gen.jpg
    """
    pairs = []
    for gt_path in gt_list:
        base = basename(gt_path).split(".")[0]

        # Construct model filename
        for m_path in model_list:
            m_name = basename(m_path)
            if base in m_name:
                pairs.append((gt_path, m_path))
                break

    return pairs


# Metric Computation
def compute_metrics(gt_path, pred_path):

    hr = Image.open(gt_path).resize((IM_W, IM_H))
    pr = Image.open(pred_path).resize((IM_W, IM_H))

    # SSIM (RGB)
    ssim = getSSIM(np.array(hr), np.array(pr))

    # PSNR on Lightness channel
    hr_L = hr.convert("L")
    pr_L = pr.convert("L")
    psnr = getPSNR(np.array(hr_L), np.array(pr_L))

    # UIQM (no reference)
    uiqm = getUIQM(np.array(pr))

    return ssim, psnr, uiqm


# MAIN PROCESS
if __name__ == "__main__":

    gt_list = load_images(GROUND_TRUTH)

    print(f"\nFound {len(gt_list)} ground truth images\n")
    results = []

    for model_name, model_dir in MODEL_DIRS.items():

        print(f"\n=== Evaluating Model: {model_name} ===")

        model_imgs = load_images(model_dir)

        if len(model_imgs) == 0:
            print(f"No images found for {model_name}, skipping...")
            continue

        pairs = match_pairs(gt_list, model_imgs, model_name)

        print(f"Matched {len(pairs)} image pairs")

        if len(pairs) == 0:
            print("❌ No matching images — check filenames.")
            continue

        SSIM_vals, PSNR_vals, UIQM_vals = [], [], []

        for gt, pred in pairs:
            ssim, psnr, uiqm = compute_metrics(gt, pred)
            SSIM_vals.append(ssim)
            PSNR_vals.append(psnr)
            UIQM_vals.append(uiqm)

        # Store summary
        results.append([
            model_name,
            np.mean(SSIM_vals), np.std(SSIM_vals),
            np.mean(PSNR_vals), np.std(PSNR_vals),
            np.mean(UIQM_vals), np.std(UIQM_vals)
        ])

        print(f"SSIM: {np.mean(SSIM_vals):.4f}")
        print(f"PSNR: {np.mean(PSNR_vals):.4f}")
        print(f"UIQM: {np.mean(UIQM_vals):.4f}")

    # Save to CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "SSIM_mean", "SSIM_std",
            "PSNR_mean", "PSNR_std",
            "UIQM_mean", "UIQM_std"
        ])
        writer.writerows(results)

    print("\nMetrics saved to:", OUTPUT_CSV)
    print("\nEvaluation Complete!\n")
