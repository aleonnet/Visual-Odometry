# -*- coding: utf-8 -*-
"""
Three-D generic tools
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt

# lucas-kanade parameters
lk_params = dict( winSize  = (21,21),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

OUT_DIR = None

###############################################################################
# projection model functions
###############################################################################
def GetTfromPar(par):
    """use the Rodrigues formula to convert from six parameters camera pose to 3x4 matrix"""
    m, j = cv2.Rodrigues(par[0:3])
    retv = np.zeros((3, 4))
    retv[:3, :3], retv[:, 3] = m, par[3:]
    return retv

def GetParFromT(T):
    """use the Rodrigues formula to convert from 3x4 matrix to 
    the six parameters camera pose"""
    p, j = cv2.Rodrigues(T[:3,:3])
    retv = np.zeros(6)
    retv[:3], retv[3:] = p.ravel(), T[:, 3]
    return retv

def CamToWld(vx, vy, KI):
    """use the inverse camera matrix to convert from pixel coordinates to wld coords"""
    mat = np.ones((3, len(vx)), dtype=np.float32)
    mat[0, :], mat[1, :] = vx.ravel(), vy.ravel()
    mat = np.dot(KI, mat)
    return mat[0,:], mat[1,:]

def CamToWld2(pv, KI):
    mat = np.ones((3, pv.shape[0]), dtype=np.float32)
    mat[0, :], mat[1, :] = pv[:,0], pv[:,1]
    mat = np.dot(KI, mat)
    return mat[0:2,:]


def WldToCam(vX, vY, vZ, K):
    """use the camera matrix to convert from world coords to pixel coordinates"""
    mat = np.ones((3, len(vX)), dtype=np.float32)
    mat[0, :], mat[1, :], mat[2, :] = vX, vY, vZ
    mat = np.dot(K, mat)
    return mat[0,:]/mat[2,:], mat[1,:]/mat[2,:]

def ProjectImgPoints(vx, vy, vz, T, KI, K):
    """Take points from one image and project in another"""

    # image to world
    vX, vY = CamToWld(vx, vy, KI)
    mat = np.ones((4, len(vx)), dtype=np.float32)
    mat[0, :], mat[1, :], mat[2, :] = vX*vz, vY*vz, vz

    # reference change (apply T) and reproject 
    mat = np.dot(T, mat)
    rx, ry = WldToCam(mat[0], mat[1], mat[2], K)
    return rx, ry

def PtsToWld(vx, vy, vz, Twc, KI):
    vX, vY = CamToWld(vx, vy, KI)
    mat = np.ones((4, len(vx)), dtype=np.float32)
    mat[0, :], mat[1, :], mat[2, :] = vX*vz, vY*vz, vz

    # reference change (apply T)
    mat = np.dot(Twc, mat)
    return mat.T

# Take world points and project into an image
def ProjectWldPoints(vX, vY, vz, T, K, getz=False):

    # world mat
    mat = np.ones((4, len(vX)))
    #mat[0, :], mat[1, :], mat[2, :] = vX*vz, vY*vz, vz
    mat[0, :], mat[1, :], mat[2, :] = vX, vY, vz

    # reference change (apply T) and reproject 
    mat = np.dot(T, mat)
    rx, ry = WldToCam(mat[0], mat[1], mat[2], K)
    if getz:
        return rx, ry, mat[2]
    else:
        return rx, ry

def Compose(T_ab, T_bc):
    """Merge camera transitions from 'a' to 'b' and 'b' to 'c' """
    T_ac = np.zeros((3, 4), dtype=np.float32)
    T_ac[:, 3] = T_ab[:, 3] + np.dot(T_ab[0:3, 0:3], T_bc[:, 3])
    T_ac[0:3, 0:3] = np.dot(T_ab[0:3, 0:3], T_bc[0:3, 0:3])
    return T_ac


def Invert(T_ab):
    """Invert camera transitions from 'a' to 'b'"""
    T_ba = np.zeros((3, 4), dtype=np.float32)
    T_ba[0:3, 0:3] = np.linalg.inv(T_ab[0:3, 0:3])
    T_ba[:, 3] = -1*np.dot(T_ba[0:3, 0:3], T_ab[:, 3])
    return T_ba


def ProjDif(measure, par):
    """Calculate projection error"""
    xk, yk, zk, xf, yf, KI, K = measure
    T = GetTfromPar(par)
    vxfp, vyfp = ProjectImgPoints(xk, yk, zk, T, KI, K)
    dl = np.abs(xf-vxfp) + np.abs(yf-vyfp)
    return dl
        
###############################################################################
# tracking functions
###############################################################################

def TrackPts(vx, vy, vz, im0, im1, maxDsqr=5):
    """ use lucas-kanade algorithm to track points from img_0->img_1, and back
     from img_1->img_0, and return only the points that were backtracted correctly"""

    # track img_0->img_1, and remove bad results
    p00 = np.hstack((vx.reshape(-1, 1), vy.reshape(-1, 1)))
    p1, st, err = cv2.calcOpticalFlowPyrLK(im0, im1, p00, None, **lk_params)
    if st is None:
        return [], [], None
    st = st.reshape(-1)

    #print len(p00), len(p1), len(vz), len(st)
    
    p00, p1, vz = p00[st==1], p1[st==1], vz[st==1]
    
    # then back-track the good results, and remove bad guys
    p0, st, err = cv2.calcOpticalFlowPyrLK(im1, im0, p1, None, **lk_params)
    if st is None:
        return [], [], None
    st = st.reshape(-1)
    err = err.reshape(-1)
    valid = (st==1) #& (err < 4)
    p00, p0, p1, vz, err = p00[valid], p0[valid], p1[valid], vz[valid], err[valid]
    
    # finally remove points that ended farther then max square distance
    dist = np.array([(x0-x1)**2+(y0-y1)**2 for (x0, y0), (x1, y1) in zip(p00, p0)])
    valid = (dist<maxDsqr)# & (err < 10)
    p0, p1, vz = p0[valid], p1[valid], vz[valid]
    return p0, p1, vz


def TrackWarp(vx, vy, vz, im0, im1, maxDsqr=5):
    """track features in affine warped images """

    global _gTW; _gTW = (vx, vy, vz, im0, im1, maxDsqr); #assert 0
    #(vx, vy, vz, im0, im1, maxDsqr) = _gTW 
    B = cv2.estimateRigidTransform(im1, im0, False)
    if B is None:
        B = np.eye(2, 3, dtype=np.float32)
    wim = cv2.warpAffine(im1, B, (im1.shape[1], im1.shape[0]))
    p0, p1, vz = TrackPts(vx, vy, vz, im0, wim, maxDsqr)
    if vz is None:
        return None, None, []
    vx1, vy1 = p1[:, 0], p1[:, 1]
    iB = cv2.invertAffineTransform(B)
    mb = np.vstack((vx1, vy1, np.ones(len(vx1))))
    p1 = (np.dot(iB, mb)).T
    #p1 = (np.dot(B, mb)).T
    return p0, p1, vz


def WarpImgT(img, T, z, hv, hh, zImg):
    """ warp image using a 4x3 matrix - note this is a test function, too slow
    to be used for real time algorithms"""
    my, mx = img.shape
    retImg, retDep = np.zeros(img.shape, dtype=img.dtype), np.zeros(img.shape, dtype=np.float32)
    for x in range(mx):
        for y in range(my):
            z = zImg[y, x]
            v = np.array([(x-hh)*z, (y-hv)*z, z, 1]).reshape((4, 1))
            vx, vy, vz = np.dot(T, v).ravel()
            vx, vy = vx/vz, vy/vz
            vx, vy = int(vx+hh), int(vy+hv)
            vx, vy = max(0, min(mx-1, vx)), max(0, min(my-1, vy))
            #retImg[y, x] = img[vy, vx]
            if retImg[vy, vx] < 1 or retDep[vy, vx] >= z:
                retImg[vy, vx] = img[y, x]
                retDep[vy, vx] = z
    return retImg


def GetZ(vx0, vy0, vx1, vy1, T_10, KI):
    """ get depth of features"""
    X0, Y0, A = np.dot(KI, np.vstack((vx0, vy0, np.ones(len(vx0)))))
    X1, Y1, A = np.dot(KI, np.vstack((vx1, vy1, np.ones(len(vx1)))))

    pts1 = np.zeros((1, len(X0), 2))
    pts1[0, :, 0] = X0
    pts1[0, :, 1] = Y0
    pts2 = np.zeros((1, len(X0), 2))
    pts2[0, :, 0] = X1
    pts2[0, :, 1] = Y1

    p = cv2.triangulatePoints(T_10, np.eye(3,4), pts2, pts1)
    return p[2]/p[3]

def threeViewTriangulation(vxc, vyc, vxp, vyp, vxn, vyn, P_nc, P_pc, K, KI, ndiv=15):
    T_nc, T_pc = GetTfromPar(P_nc), GetTfromPar(P_pc)
    vz0 = GetZ(vxc, vyc, vxn, vyn, T_nc, KI)
    vz1 = GetZ(vxc, vyc, vxp, vyp, T_pc, KI)
    rgLst = np.zeros((ndiv, len(vxc)))
    for i, (zst, zed) in enumerate(zip(vz0, vz1)):
        rgLst[:, i] = np.linspace(zst, zed, ndiv)

    zl = np.zeros((ndiv, len(vxc)))
    for i, r in enumerate(rgLst):
        z01 = ProjDif((vxc, vyc, r, vxp, vyp, KI, K), P_pc)
        z21 = ProjDif((vxc, vyc, r, vxn, vyn, KI, K), P_nc)
        zl[i] = z01+z21
    am = np.argmin(zl, axis=0)

    minz = np.array([rg[am[i]] for i, rg in enumerate(rgLst.T)])
    return minz

def SimpleLocate(rxc, ryc, rzc, rxn,ryn, K, KI):
    """ locate function using opencv's solvePnP    """

    X0, Y0, A = np.dot(KI, np.vstack((rxc, ryc, np.ones(len(rxc)))))
    X0, Y0 = X0*rzc, Y0*rzc
    pts1 = (np.vstack((X0, Y0, rzc)).T).reshape(1, -1, 3)
    pts2 = (np.vstack((rxn, ryn)).T).reshape(1, -1, 2)
    pts1, pts2 = pts1.astype(np.float32), pts2.astype(np.float32)
    ok, r, t = cv2.solvePnP(pts1, pts2, K, None)
    tP_nc = (np.hstack((r.ravel(), t.ravel())))
    tT_nc = GetTfromPar(tP_nc)
    return tP_nc, tT_nc


###############################################################################
# plotting functions
###############################################################################

def PutPoints(imOut, vx, vy, vz):
    if len(vx)>0:
        if len(vz)==0:
            vz = np.ones(len(vx))
        c = np.clip(vz, 0.2, 5)
        c = ((np.log(c)-np.log(c.min()))*(
        255/(np.log(c.max())-np.log(c.min())+1E-8))).astype(np.int)
        for i, (x, y) in enumerate(zip(vx, vy)):
            color = (255-c[i], 0, c[i])
            pt0, pt1 = (int(x)-4, int(y)-4), (int(x)+4, int(y)+4)
            cv2.rectangle(imOut, pt0, pt1, color, 2)

def PutLines(imOut, vxc, vyc, vxn, vyn, vz):
    if len(vxc)>0:
        if len(vz)==0:
            vz = np.ones(len(vxc))
        c = np.clip(vz, 0.2, 5)
        c = ((np.log(c)-np.log(c.min()))*(
        255/(np.log(c.max())-np.log(c.min())+1E-8))).astype(np.int)
        for i, (xc, yc, xn, yn) in enumerate(zip(vxc, vyc, vxn, vyn)):
            color = (255-c[i], 0, c[i])
            pt0, pt1 = (int(xc)-1, int(yc)-1), (int(xc)+1, int(yc)+1)
            cv2.rectangle(imOut, pt0, pt1, color)
            pt0, pt1 = (int(xc), int(yc)), (int(xn), int(yn))
            cv2.line(imOut, pt0, pt1, color, thickness=2)

            #color = (255, 0, 0)
            #cv2.circle(imOut, (int(x), int(y)), 3, color, thickness=-1)

def PutMessages(imOut, msgs):
    vr, hr = imOut.shape[0], imOut.shape[1]
    for i, msg in enumerate(msgs):
        pos = (hr/3, vr-15*i-15)
        cv2.putText(imOut, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0))
    
def PrintImgPtsMsg(img, vx=[], vy=[], vz=[], idx=1, msgs=[], outPath=None, save=1, show=False):

    # print points and messages:
    imOut = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    PutPoints(imOut, vx, vy, vz)
    PutMessages(imOut, msgs)

    if outPath is None:
        outPath = OUT_DIR

    if save == 0:
        plt.figure(idx); plt.imshow(imOut),plt.show()
    else:
        b,g,r = cv2.split(imOut)       # get b,g,r
        imOut = cv2.merge([r,g,b])     # switch it to rgb
        if show:
            cv2.imshow("img", imOut)
        else:
            cv2.imwrite(outPath %(idx), imOut)


def PrintImgPtsDelta(img, vxc, vyc, vxn, vyn, vz, idx=1, msgs=[], outPath=None, save=1, show=False):

    # print points and messages:
    imOut = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    PutPoints(imOut, vxc, vyc, vz)
    PutLines(imOut, vxc, vyc, vxn, vyn, vz)
    PutMessages(imOut, msgs)

    if outPath is None:
        outPath = OUT_DIR

    if save == 0:
        plt.figure(idx); plt.imshow(imOut),plt.show()
    else:
        b,g,r = cv2.split(imOut)       # get b,g,r
        imOut = cv2.merge([r,g,b])     # switch it to rgb
        if show:
            cv2.imshow("img", imOut)
        else:
            cv2.imwrite(outPath %(idx), imOut)
