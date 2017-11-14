# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:53:10 2016

@author: irigon
"""

import numpy as np
import cv2
import tools_3d as t3d
import conf_3d as c3d
import os

(major, minor, _) = cv2.__version__.split(".")

# ###############################################################
# load files
# ###############################################################
base = "C:/kitti/dataset/"

if 1:
    trackNumber = 3
    #poseFname = "%s%s%02d%s" % (base, "poses/", trackNumber, ".txt")
    #poses = np.loadtxt(poseFname)
    calibFname = "%s%s%02d%s" % (base, "sequences/", trackNumber, "/calib.txt")
    calib = np.loadtxt(calibFname, usecols = range(1,13))

K = calib[0].reshape(3,4)
K = K[:3, :3].astype(np.float32)
KI = np.linalg.inv(K)
c3d.K = K
c3d.KI = KI
t3d.K = K


def SetTrackNumber(tn):
    """setup folders and calibration matrix"""
    global trackNumber, poseFname, poses, K, KI
    trackNumber = tn
    poseFname = "%s%s%02d%s" % (base, "poses/", trackNumber, ".txt")
    if os.path.isfile(poseFname):
        poses = np.loadtxt(poseFname)
    else:
        imgDir = "%s%s%02d%s" % (base, "sequences/", trackNumber, "/image_0/")
        lst = os.listdir(imgDir)
        poses = np.zeros((len(lst), 12))
        poses[:, 0], poses[:, 5], poses[:, 10] = 1, 1, 1
    calibFname = "%s%s%02d%s" % (base, "sequences/", trackNumber, "/calib.txt")
    calib = np.loadtxt(calibFname, usecols = range(1,13))
    K = calib[0].reshape(3,4)
    K = K[:3, :3]
    KI = np.linalg.inv(K)
    c3d.K = K
    c3d.KI = KI
    t3d.K = K


def GetImg(imIdx):
    """load kitti image"""
    imgsFname = "%s%s%02d%s%06d%s" % (base, "sequences/", trackNumber, "/image_0/", imIdx, ".png")
    if major == '3':
        return cv2.imread(imgsFname, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(imgsFname, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def GetImg_1(imIdx):
    imgsFname = "%s%s%02d%s%06d%s" % (base, "sequences/", trackNumber, "/image_1/", imIdx, ".png")
    return cv2.imread(imgsFname, cv2.IMREAD_GRAYSCALE)


def GetPlst():
    """get list of camera poses (from ground truth)"""
    plst = np.zeros((len(poses)-1, 6))
    for i in range(len(poses)-1):
        T_w_i, T_w_ip = poses[i].reshape(3, 4), poses[i+1].reshape(3, 4)
        T_ip_i = t3d.Compose(t3d.Invert(T_w_ip), T_w_i)
        plst[i] = (t3d.GetParFromT(T_ip_i))
    return plst

def fixPl(pl, spdl=None):
    npl = np.copy(pl)
    for i in range(1, len(pl)):
        if abs(pl[i, 1])>0.08 or abs(pl[i, 3])>0.2 or abs(pl[i, 1] - pl[i-1, 1])>0.03 or abs(pl[i, 1] - npl[i-1, 1])>0.03:
            npl[i] = npl[i-1]
            npl[i, 1] = npl[i, 1]*0.95
        if pl[i, 5]> (-0.03):
            npl[i] = np.zeros(6)
        if spdl is not None:
            if abs(pl[i, 5] - spdl[i])> 0.3:
                npl[i, 5] = spdl[i]
    return npl

def PosesToWorld(poses):
    """change reference poses from position (n-1)->(n) to (0)->(n)"""
    wposes  = np.zeros((len(poses)+1, 6))
    t0 = np.eye(3, 4)
    for i, p in enumerate(poses):
        wposes[i] = t3d.GetParFromT(t0)
        t0 = t3d.Compose(t0, t3d.Invert(t3d.GetTfromPar(p)))
    wposes[-1] = t3d.GetParFromT(t0)
    return wposes

