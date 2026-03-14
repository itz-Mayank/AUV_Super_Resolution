import os
import datetime
import numpy as np
import cv2
import natsort

from CloseDepth import closePoint
from F_stretching import StretchingFusion
from MapFusion import Scene_depth
from MapOne import max_R
from MapTwo import R_minus_GB
from blurrinessMap import blurrnessMap
from getAtomsphericLightFusion import ThreeAtomsphericLightFusion
from getAtomsphericLightOne import getAtomsphericLightDCP_Bright
from getAtomsphericLightThree import getAtomsphericLightLb
from getAtomsphericLightTwo import getAtomsphericLightLv
from getRGbDarkChannel import getRGB_Darkchannel
from getRefinedTransmission import Refinedtransmission
from getTransmissionGB import getGBTransmissionESt
from getTransmissionR import getTransmission
from global_Stretching import global_stretching
from sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')

starttime = datetime.datetime.now()

# -------------------------------------------------------------------
# CORRECT INPUT & OUTPUT PATHS FOR UFO-120
# -------------------------------------------------------------------

# UFO-120 TEST LR images
folder = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Code\Data\UFO-120\TEST\lrd"

# Output folder for IFM results
output_folder = r"C:\Users\Mayank Meghwal\Desktop\Reserch Papers AUV\AUV\Results\IFM"
os.makedirs(output_folder, exist_ok=True)

files = natsort.natsorted(os.listdir(folder))

# -------------------------------------------------------------------
# MAIN PROCESSING LOOP
# -------------------------------------------------------------------

for i, file in enumerate(files):

    filepath = os.path.join(folder, file)

    if not os.path.isfile(filepath):
        continue

    print(f"Processing file: {file}")

    img = cv2.imread(filepath)

    blockSize = 9
    n = 5

    RGB_Darkchannel = getRGB_Darkchannel(img, blockSize)
    BlurrnessMap = blurrnessMap(img, blockSize, n)

    # Atmospheric Light estimation
    AL1 = getAtomsphericLightDCP_Bright(RGB_Darkchannel, img, percent=0.001)
    AL2 = getAtomsphericLightLv(img)
    AL3 = getAtomsphericLightLb(img, blockSize, n)
    AtomsphericLight = ThreeAtomsphericLightFusion(AL1, AL2, AL3, img)

    # Depth map generation
    R_map = max_R(img, blockSize)
    mip_map = R_minus_GB(img, blockSize, R_map)

    d_R = 1 - StretchingFusion(R_map)
    d_D = 1 - StretchingFusion(mip_map)
    d_B = 1 - StretchingFusion(BlurrnessMap)

    d_n = Scene_depth(d_R, d_D, d_B, img, AtomsphericLight)
    d_n_stretching = global_stretching(d_n)

    d_0 = closePoint(img, AtomsphericLight)
    d_f = 8 * (d_n + d_0)

    # Transmission map
    transmissionR = getTransmission(d_f)
    transmissionB, transmissionG = getGBTransmissionESt(transmissionR, AtomsphericLight)
    transmissionB, transmissionG, transmissionR = Refinedtransmission(
        transmissionB, transmissionG, transmissionR, img
    )

    # Recover scene radiance (final IFM enhanced image)
    sceneRadiance = sceneRadianceRGB(img, transmissionB, transmissionG, transmissionR, AtomsphericLight)

    out_path = os.path.join(output_folder, f"IFM_{i}.jpg")
    cv2.imwrite(out_path, sceneRadiance)

# -------------------------------------------------------------------
# END LOG
# -------------------------------------------------------------------

Endtime = datetime.datetime.now()
print("Time Taken:", Endtime - starttime)
