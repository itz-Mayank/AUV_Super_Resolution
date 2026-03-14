# -*- coding: utf-8 -*-
"""
Fusion Script for DirectGAN + IUFusion and IFM + UCM
Updated with correct paths and safety checks.
"""

import os
import cv2
import natsort
import numpy as np

np.seterr(over='ignore')

# --------------------------------------------------------------------
# SELECT FUSION MODE
# Options: "IFM_UCM" or "GAN_IU"
# --------------------------------------------------------------------

fusion_mode = "GAN_IU"       # change to "IFM_UCM" if needed
# fusion_mode = "IFM_UCM"       # change to "IFM_UCM" if needed

# --------------------------------------------------------------------
# 1. IFM + UCM FUSION
# --------------------------------------------------------------------

if fusion_mode == "IFM_UCM":

    base_folder = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Results\Fusion\Zip"

    path_ifm = os.path.join(base_folder, "1IFM")
    path_ucm = os.path.join(base_folder, "1UCM")
    output_folder = os.path.join(base_folder, "Fusion")

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(path_ifm):
        print("Missing IFM folder:", path_ifm)

    if not os.path.exists(path_ucm):
        print("Missing UCM folder:", path_ucm)

    files_ifm = natsort.natsorted(os.listdir(path_ifm)) if os.path.exists(path_ifm) else []
    files_ucm = natsort.natsorted(os.listdir(path_ucm)) if os.path.exists(path_ucm) else []

    print("IFM files:", len(files_ifm))
    print("UCM files:", len(files_ucm))

    for f_ifm, f_ucm in zip(files_ifm, files_ucm):

        file_ifm = os.path.join(path_ifm, f_ifm)
        file_ucm = os.path.join(path_ucm, f_ucm)

        if not (os.path.isfile(file_ifm) and os.path.isfile(file_ucm)):
            continue

        img1 = cv2.imread(file_ifm)
        img2 = cv2.imread(file_ucm)

        fused = cv2.addWeighted(img1, 0.50, img2, 0.55, 0.0)
        cv2.imwrite(os.path.join(output_folder, f_ifm), fused)

    print("IFM + UCM fusion complete.")

# --------------------------------------------------------------------
# 2. DirectGAN + IUFusion FUSION
# --------------------------------------------------------------------

elif fusion_mode == "GAN_IU":

    base_folder = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Results\ALL"

    path_gan = os.path.join(base_folder, "DirectGAN")
    path_iu = os.path.join(base_folder, "IUFusion")
    output_folder = os.path.join(base_folder, "FuseFusionGAN")

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(path_gan):
        print("Missing DirectGAN folder:", path_gan)

    if not os.path.exists(path_iu):
        print("Missing IUFusion folder:", path_iu)

    files_gan = natsort.natsorted(os.listdir(path_gan)) if os.path.exists(path_gan) else []
    files_iu = natsort.natsorted(os.listdir(path_iu)) if os.path.exists(path_iu) else []

    print("DirectGAN files:", len(files_gan))
    print("IUFusion files:", len(files_iu))

    for f_gan, f_iu in zip(files_gan, files_iu):

        file_gan = os.path.join(path_gan, f_gan)
        file_iu = os.path.join(path_iu, f_iu)

        if not (os.path.isfile(file_gan) and os.path.isfile(file_iu)):
            continue

        img1 = cv2.imread(file_gan)
        img2 = cv2.imread(file_iu)

        fused = cv2.addWeighted(img1, 0.50, img2, 0.50, 0.0)
        cv2.imwrite(os.path.join(output_folder, f_gan), fused)

    print("DirectGAN + IUFusion fusion complete.")

else:
    print("Invalid fusion mode selected.")
