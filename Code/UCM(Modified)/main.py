import os
import numpy as np
import cv2
import natsort
import xlwt
import datetime

from color_equalisation import RGB_equalisation
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

starttime = datetime.datetime.now()

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/NonPhysical/UCM"
folder = "C:\\Users\\Mayank Meghwal\\Desktop\\Reserch Papers AUV\\AUV\\Code\\Data\\UFO-120\\TEST"

path = folder + "\\lrd"
files = os.listdir(path)
files = natsort.natsorted(files)

output_folder = "C:\\Users\\Mayank Meghwal\\Desktop\\Reserch Papers AUV\\AUV\\Results\\UCM\\"
os.makedirs(output_folder, exist_ok=True)

for i in range(len(files)):
    file = files[i]
    filepath = path + "\\" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********', file)
        img = cv2.imread(filepath)

        sceneRadiance = RGB_equalisation(img)
        sceneRadiance = stretching(sceneRadiance)
        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance = sceneRadianceRGB(sceneRadiance)

        cv2.imwrite(output_folder + prefix + "_UCM.jpg", sceneRadiance)

endtime = datetime.datetime.now()
time = endtime - starttime
print('time', time)
