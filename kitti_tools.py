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



def MakeKeyFrame(n):
    # get current and prev images
    assert n > 0
    global fts, kFrameList
    imc, imp = GetImg(n), GetImg(n-1)
    T_wc, T_wp = poses[n].reshape(3, 4), poses[n-1].reshape(3, 4)
    #T_wc, T_wp = poses[0].reshape(3, 4), poses[0].reshape(3, 4)
    vxc, vyc = t3d.ClearPointsToTrack(imc, [], [], maxPts=500, mindist=15)
    p0, p1, vz0 = t3d.TrackWarp(vxc, vyc, np.ones(len(vxc)), imc, imp, maxDsqr=5)
    (vxc, vyc ), (vxp, vyp) = p0.T, p1.T
    fts = np.zeros(len(vxc), dtype=object)
    for i, (xc, yc) in enumerate(zip(vxc, vyc)):
        fts[i]  = t3d.SharedFeat(xc, yc, 1.0, 0, 10)
    lst = np.arange(len(vxc))
    t_wk = T_wc
    t_kw = t3d.Invert(T_wc)
    p_kw = t3d.GetParFromT(T_wc)
    kf = t3d.SfKeyFrame(imc, vxc, vyc, lst, np.copy(lst), p_kw, t_kw, t_wk)
    kFrameList = [kf]
    #corr, mFtIdx = t3d.MatchFeats(im1, kf, fts[kf.ftIdxs])
    mFtIdx = np.arange(len(vxc)) #kf.ftIdxs[mFtIdx]
    vxk, vyk, vxf, vyf = vxc, vyc, vxp, vyp # corr
    zf = t3d.GetFeatDepth(vxf, vyf, t3d.Invert(T_wp), mFtIdx, fts, kFrameList, KI, update=True)
    zl = np.array([f.z for f in fts])
    #t3d.PrintSkFr(imc, vxc, vyc, zl/20, n, p, 0, 0.5, kFrameList, c3d.OUTPUT, save=1, show=False)


# #############################################################################
# get road plane: (not used)
# #############################################################################
trapMask = np.ones((372-214,830-220), dtype=np.uint8)
for y in range(158):
    dx = 518-220
    for x in range(518-220):
        if x < ((158-y)/158.0)*dx:
            trapMask[y, x] = 0
    dx = 830-654
    for x in range(830-654):
        if x > (y/158.0)*dx:
            trapMask[y, x+654-220] = 0

def IsPointInTrap(px, py):
    
    if py < 214 or px < 220 or px >= 830:
        return False
    if trapMask[py-214, px-220] == 0:
        return False
    return True

def isVecIntTrap(vx, vy):
    retVec = np.zeros(len(vx), dtype=np.bool)
    for i, (px, py) in enumerate(zip(vx, vy)):
        retVec[i] = IsPointInTrap(px, py)
    return retVec
    

def GetPtsInSides(img, nFts=500):
    cutImg = img[:,0:480]
    vx0, vy0 = t3d.ClearPointsToTrack(cutImg, [], [], maxPts=nFts, mindist=10)
    pti = isVecIntTrap(vx0, vy0)
    vx0, vy0 = vx0[pti==False], vy0[pti==False]
    cutImg = img[:,700:]
    vx1, vy1 = t3d.ClearPointsToTrack(cutImg, [], [], maxPts=nFts, mindist=10)
    vx1 = vx1+700
    pti = isVecIntTrap(vx1, vy1)
    vx1, vy1 = vx1[pti==False], vy1[pti==False]
    return np.hstack((vx0, vx1)), np.hstack((vy0, vy1))
    

def GetFeatsInTrapezoid(img, nfeats=200, upper=False):
    global __gImg; __gImg=img
    cutImg = img[212:370,220:830]
    print cutImg.shape, trapMask.shape
    cutImg = cutImg*trapMask
    if upper:
        cutImg[100:] = 0
    #t3d.PrintSkFr(cutImg, [], [], [], 2, p, 0, 0.5, kFrameList, c3d.OUTPUT, save=0, show=False)
    vx0, vy0 = t3d.ClearPointsToTrack(cutImg, [], [], maxPts=nfeats, mindist=5)
    return vx0+220, vy0+214


def GetTrapPlane(n):
    im0, im1 = GetImg(n), GetImg(n+1)
    T_w0, T_w1  = poses[n].reshape(3, 4), poses[n+1].reshape(3, 4)
    vxc, vyc = GetFeatsInTrapezoid(im0)
    p0, p1, vz0 = t3d.TrackWarp(vxc, vyc, np.ones(len(vxc)), im0, im1, maxDsqr=5)
    (vx0, vy0), (vx1, vy1) = p0.T, p1.T
    T_10 = t3d.Compose(t3d.Invert(T_w1), T_w0)
    vz0 = t3d.GetZ(vx0, vy0, vx1, vy1, T_10, KI)
    vz1 = t3d.GetZ(vx1, vy1, vx0, vy0, t3d.Invert(T_10), KI)

    p = np.zeros(6)
    #t3d.PrintSkFr(im0, vx0, vy0, vz0/10, 2, p, 0, 0.5, [], c3d.OUTPUT, save=1, show=False)    
    #t3d.PrintSkFr(im1, vx1, vy1, vz1/10, 3, p, 0, 0.5, [], c3d.OUTPUT, save=0, show=False)    
    X0, Y0, A = np.dot(KI, np.vstack((vx0, vy0, np.ones(len(vx0)))))
    X0, Y0 = X0*vz0, Y0*vz0

    #A = np.vstack((vx0, vy0, vz0, np.ones(len(vx0))))
    A = np.vstack((X0, Y0, vz0, np.ones(len(vx0))))
    u, d, v = np.linalg.svd(np.dot(A, A.T))
    plane = (u[:, 3]).reshape(4, 1)
    e = np.dot(A.T, plane)
    return plane

#pl = plane/plane[2]
#pl[2] = 0
##np.dot(A[:, 4:5].T, pl)
#d = (-1*np.dot(A.T, pl)).ravel()


#plane = GetTrapPlane(15)
# use only points in trap to get position
def GetTranslation(plane, st, ed):
    T_w_st = poses[st].reshape(3, 4)
    nP_FW = np.zeros(6)
    pl = plane/plane[2]
    pl[2] = 0
    acce, accT = 0.0, np.eye(3,4)
    for n in range(st,ed):
        im0, im1 = GetImg(n), GetImg(n+1)
        T_w0, T_w1  = poses[n].reshape(3, 4), poses[n+1].reshape(3, 4)
        vxc, vyc = GetFeatsInTrapezoid(im0)
        p0, p1, vz0 = t3d.TrackWarp(vxc, vyc, np.ones(len(vxc)), im0, im1, maxDsqr=5)
        (vx0, vy0), (vx1, vy1) = p0.T, p1.T
        T_10 = t3d.Compose(t3d.Invert(T_w1), T_w0)
        p_10 = t3d.GetParFromT(T_10)
        #vz0 = t3d.GetZ(vx0, vy0, vx1, vy1, T_10, KI)
        #vz1 = t3d.GetZ(vx1, vy1, vx0, vy0, t3d.Invert(T_10), KI)
        A = np.vstack((vx0, vy0, np.zeros(len(vx0)), np.ones(len(vx0))))
        vz = (-1*np.dot(A.T, pl)).ravel()
    
        Xw, Yw, A = np.dot(KI, np.vstack((vx0, vy0, np.ones(len(vx0)))))
        Xw, Yw = Xw*vz, Yw*vz
        mw = (vx1, vy1, Xw, Yw, vz)
        p0 = nP_FW*1.2#np.hstack((np.zeros(5), [-1.2]))
        #prevP = nP_FW
        #p0 = np.hstack((np.zeros(5), [-0.2]))
        nP_FW, err = t3d.LevMar(mw, t3d.ProjDifVarSharedFeatsWld, p0, dump=1, conv=1, maxIt=6)
        ntry=0
        while ntry<5 and err>2:
            d = t3d.ProjDifVarSharedFeatsWld(mw, nP_FW)
            maxd = d.mean()+ 1.6*d.std()
            good, bad = d<=maxd, d>maxd
            (vx1, vy1, Xw, Yw, vz) = (vx1[good], vy1[good], Xw[good], Yw[good], vz[good])
            mw = (vx1, vy1, Xw, Yw, vz)
            nP_FW, err = t3d.LevMar(mw, t3d.ProjDifVarSharedFeatsWld, nP_FW, dump=1, conv=1, maxIt=6)
            ntry = ntry+1
        
        #nP_FW, err = t3d.LevMar(mw, t3d.ProjDifVarSharedFeatsWld, nP_FW, dump=0.2, conv=1, maxIt=6)
        accT = t3d.Compose(accT, t3d.GetTfromPar(nP_FW))
        #print("%d, %.2f, %.2f, %.2f, %d" % 
        #    (n, accT[2,3] + T_w1[2,3], accT[1,3] + T_w1[1,3], err, ntry))
        acce += np.sum(np.abs(p_10 - nP_FW))

    T_ed_st = t3d.Compose(t3d.Invert(T_w1), T_w_st)
    print (accT[:,3])
    #print (T_w1[:,3])
    print (T_ed_st[:,3])
    

def GetPtsRealPos(vx0, vy0, n0, vx1, vy1, n1):
    T_w_n0, T_w_n1 = poses[n0].reshape(3, 4), poses[n1].reshape(3, 4)
    T_n1_n0 = t3d.Compose(t3d.Invert(T_w_n1), T_w_n0)
    vz0 = t3d.GetZ(vx0, vy0, vx1, vy1, T_n1_n0, KI)
    return vz0

def GetPtsRealPos2(pv0, n0, pv1, n1):
    T_w_n0, T_w_n1 = poses[n0].reshape(3, 4), poses[n1].reshape(3, 4)
    T_n1_n0 = t3d.Compose(t3d.Invert(T_w_n1), T_w_n0)
    vz0 = t3d.GetZ2(pv0, pv1, T_n1_n0, KI)
    return vz0


def Relocate(vx0, vy0, vz0, vx1, vy1):
    X0, Y0, A = np.dot(KI, np.vstack((vx0, vy0, np.ones(len(vx0)))))
    X0, Y0 = X0*vz0, Y0*vz0
    mw = (vx1, vy1, X0, Y0, vz0)
    nP_10, err = IterateLevMar(mw)
    return nP_10, err


def ReprojErr(vxc, vyc, vzc, vxp, vyp, T_pc):
    vxpp, vypp = t3d.ProjectImgPoints(vxc, vyc, vzc, T_pc, KI, K)
    d = (vxpp-vxp)**2 + (vypp-vyp)**2
    return (d)


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


    