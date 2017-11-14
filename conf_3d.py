# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:05:41 2016

@author: irigon
"""
import numpy as np
HOME = False
SUCURI = True
#V_FILE, V_START, V_STEP = "/home/irigon/Downloads/3d/chess_logitech_2.mp4", 000, 000
#V_FILE, V_START, V_STEP = "/home/irigon/Downloads/3d/lab2.mp4", 000, 000
V_FILE, V_START, V_STEP = "/home/irigon/Downloads/3d/lab3.mp4", 000, 000
#V_FILE, V_START, V_STEP = None, 000, 000
#DIST = "/home/irigon/Downloads/3d/pyOut"+os.sep
#OUTPUT = "/home/irigon/Downloads/3d/pyOut/im%02d.png"
if HOME:
    PARS_FILE = "/home/f/prj/vision/slam/kitti/corresp/%02d/pars_%03d.txt"
    PL_FILE = "/home/f/prj/vision/slam/kitti/corresp/%02d/pl.txt"
elif SUCURI:
    PARS_FILE = "C:/tmp/multiView/%02d/pars_%03d.txt"
    PL_FILE = "C:/tmp/multiView/%02d/pl.txt.txt"
    DIST = "C:/tmp/multiView/"
    OUTPUT = "C:/tmp/3d_out/im%04d.png"

SHOW=True

Fx, Fy, Cx, Cy = 400, 400, 160, 120

# camera calibration matrix (internal camera parameters)
K = np.array([[   Fx,    0.0,     Cx],
              [  0. ,     Fy,     Cy], 
              [  0. ,    0.0,    1.0]], dtype=np.float32)
KI = np.linalg.inv(K)

