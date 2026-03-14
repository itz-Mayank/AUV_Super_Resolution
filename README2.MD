# 1. RUN IFM (Modified IBLA)
python Code/IFM(main)/main.py


# 2. RUN UCM (Color Correction)
python Code/UCM(Modified)/main.py


# 3. FUSE IFM + UCM RESULTS
python Code/Fusion/Blend.py


# 4. RUN GAN-BASED ENHANCEMENT (FUnIE-GAN)
# TRAIN MODEL (optional)
python Code/FUnIEGAN/train_funiegan.py    # only if you want to retrain

# TEST MODEL (mandatory)
python Code/FUnIEGAN/test.py


# 5. CREATE LR PATCHES FOR SRDRM
# (Generates lr_2x folder)
python Code/SRDRM/create_lr2x.py


# 6. TRAIN SRDRM (2× Super-Resolution)
# Patch-based training on UFO-120
python Code/SRDRM/train_genarative_models_2x.py


# 7. TEST SRDRM (GENERATE 2× SR IMAGES)
# Outputs saved in Results/SRDRM/2x/
python Code/SRDRM/test_SR_2x.py


# 8. EVALUATE METRICS (SSIM, PSNR, UIQM)
# Compares SR images with HR ground truth
python Code/PE/measure.py


# PIPELINE COMPLETE
echo "All enhancement + SR + evaluation steps completed successfully."
