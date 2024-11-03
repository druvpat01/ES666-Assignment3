import pdb
import src
from glob import glob
import importlib
import os
import cv2
import tqdm
from src.DhruvPatel.stitcher import Panaroma
import numpy as np
from natsort import natsorted



# ### Change path to images here
path = 'Images{}*'.format(os.sep)  # Use os.sep, Windows, linux have different path delimiters
# ###

os.makedirs('./results/', exist_ok=True)

img = glob('./Images/I1/STA_0031.JPG')

inst = Panaroma()

match = []

for impaths in natsorted(glob(path)):


    stitched_image, homography_matrix_list = inst.create_panaroma(path=impaths)

    # panaroma_img = inst.create_panaroma(impaths)
    # # match.append(matches)
    # # outfile =  './results/{}/{}.png'.format(impaths.split(os.sep)[-1],'panaroma')
    os.makedirs(f'./results/{impaths.split("/")[-1]}',exist_ok=True)
    cv2.imwrite(f'./results/{impaths.split("/")[-1]}/DhruvPatel_sticher.jpg', stitched_image)
    # # print(homography_matrix_list)
    # print('Panaroma saved ... @ ./results/{}.png'.format(spec.name))
    # print('\n\n')

