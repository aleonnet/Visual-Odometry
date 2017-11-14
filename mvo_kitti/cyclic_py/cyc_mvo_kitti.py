"""
Ths is the main file: it is divided in three parts: feature tracking, scale 
estimation and pose estimation. The main function runs the estimation and 
there is a graph generation in the end of the file. 
@author: irigon
"""

import numpy as np
import kitti_tools as ktt
import tools_3d as t3d
import cv2
import time
from matplotlib import pyplot as plt


MIN_PT_DIST_NEW = 10
MIN_PT_DIST_KEEP = 15
RH, RV = 1241, 376
RANSAC_TRY = 55

###########################################################################
# FEATURE TRACKING
###########################################################################

# LUCAS KANADE PARAMETERS
lk_params = dict( winSize  = (21,21),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def GetWaprMat(p0, z0, t_nc):
    """Use seed points to generate a warp matrix and its inverse"""
    xnr, ynr = t3d.ProjectImgPoints(p0[:,0], p0[:,1], z0, t_nc, ktt.KI, ktt.K)
    pnr = np.vstack((xnr, ynr)).T
    p0, pnr = p0.astype(np.float32), pnr.astype(np.float32)
    W = cv2.getPerspectiveTransform(p0, pnr)
    WI = cv2.getPerspectiveTransform(pnr, p0)
    return W, WI

def WarpPts(p, W):
    """ Use the warp matrix to move points to the warped positions"""
    ph = np.hstack((p, np.ones((len(p),1))))
    phw = np.dot(W, ph.T)
    pw = (phw[0:2]/phw[2]).T.astype(np.float32)
    return pw
   

def GetPtsHomogTwoDepths(imp, imc, imn, t_nc, pc, p0, z0, p1, z1):
    """Take images and locate features on warped versions of those images"""
    
    # get perspective transformation matrices (warp matrices and warp original points)
    W, WI = GetWaprMat(p0, z0, t_nc)
    Q, QI = GetWaprMat(p1, z1, t_nc)
    pcw = WarpPts(pc, W)
    pcq = WarpPts(pc, Q)

    # warp images and locate points on the warped images
    iwp = cv2.warpPerspective(imp, W, (imc.shape[1], imc.shape[0]))
    iwc = cv2.warpPerspective(imc, W, (imc.shape[1], imc.shape[0]))
    iqc = cv2.warpPerspective(imc, Q, (imc.shape[1], imc.shape[0]))
    pn, st0, err = cv2.calcOpticalFlowPyrLK(iwc, imn, prevPts=pcw, nextPts=None, **lk_params)
    qn, stn, err = cv2.calcOpticalFlowPyrLK(iqc, imn, prevPts=pcq, nextPts=None, **lk_params)
    ppw, st1, err = cv2.calcOpticalFlowPyrLK(imc, iwp, prevPts=pc, nextPts=None, **lk_params)
    pc0w, stn, err = cv2.calcOpticalFlowPyrLK(imn, iwc, prevPts=pn, nextPts=None, **lk_params)
    pc0q, stn, err = cv2.calcOpticalFlowPyrLK(imn, iqc, prevPts=qn, nextPts=None, **lk_params)
    pc1, stp, err = cv2.calcOpticalFlowPyrLK(iwp, imc, prevPts=ppw, nextPts=None, **lk_params)

    # calculate the distance from the original points to the back-located points
    pp = WarpPts(ppw, WI)
    pc0 = WarpPts(pc0w, WI)
    pc2 = WarpPts(pc0q, QI)
    d0 = np.sqrt((pc0[:,0]- pc[:,0])**2+(pc0[:,1]- pc[:,1])**2)
    d1 = np.sqrt((pc1[:,0]- pc[:,0])**2+(pc1[:,1]- pc[:,1])**2)
    d2 = np.sqrt((pc2[:,0]- pc[:,0])**2+(pc2[:,1]- pc[:,1])**2)
    dq = np.sqrt((pc0[:,0]- pc2[:,0])**2+(pc0[:,1]- pc2[:,1])**2)
 
    # discard non-matching points (distance greater then one pixel)
    pn[d2<d0] = qn[d2<d0]
    gd = (d0<1)*(d1<1)*(dq<1)
    gd = gd*((st0.ravel())*(st1.ravel())).astype(np.bool)
    gd = gd*(pn[:,1]<(imc.shape[0]-10))
    return pp[gd], pc[gd], pn[gd]


def GetPtsTwoDepths(imp, imc, imn, t_nc, scd):
    """find points in warped images (different fro ground points)"""

    # get points on the ground
    pc = cv2.goodFeaturesToTrack(imc, 600, qualityLevel=0.01, minDistance=15, blockSize=3)[:, 0]
    gnd = ((370- pc[:,1])**2+((600- pc[:,0])**2)/8) < (190**2)
    ngnd = np.logical_not(gnd)
    
    # use default ground position to warp images and track points
    p0 = np.array([[500, 220], [700, 220], [400, 300], [800, 300]])
    z0 = get_z_plane(p0[:,0], p0[:,1], scd)
    p1 = np.copy(p0)
    z1 = z0*(min(0.8, max(0.5, -1*t_nc[2,3]/8+0.5)))
    pp0, pc0, pn0 = GetPtsHomogTwoDepths(imp, imc, imn, t_nc, pc[gnd], p0, z0, p1, z1)   
 
    # find rest of points with two constant depths
    p0 = np.array([[200, 50], [200, 200], [1000, 50], [1000, 200]])
    z0 = np.ones(len(p0))*15
    z1 = np.ones(len(p0))*60
    pp1, pc1, pn1 = GetPtsHomogTwoDepths(imp, imc, imn, t_nc, pc[ngnd], p0, z0, p0, z1)    
    
    pp1, pc1, pn1 = np.vstack((pp0,pp1)), np.vstack((pc0,pc1)), np.vstack((pn0,pn1))
    return pp1, pc1, pn1


###########################################################################
# SCALE
###########################################################################

def get_z_plane(vxc, vyc, scd):
    """distance from camera for ground points based on image position"""
    vzc = ktt.K[0,0]*1.65/(vyc-scd)
    return vzc

def turnSig(arr, m, a):
    """sigmoid function (smoothing) """
    return 1/(1+np.exp(-1*(arr-m)/a))


def difCrop(vxc, vyc, vxn, vyn, imc, imn, nP_nc):
    """absolute differnce  (SAD) along epipolar line"""
    
    # window size and ground points
    wsz=6
    gd = (((vxc-600)**2)/4 + (vyc-375)**2) < (160*160)
    gd = gd*(vyc<imc.shape[0]-wsz/2-2)
    if gd.sum() == 0:
        return np.zeros(len(vxc))

    # project points with different depth to find position along ep. and normalize
    zc = t3d.GetZ(vxc, vyc, vxn, vyn, t3d.GetTfromPar(nP_nc), ktt.KI)
    xpr, ypr = t3d.ProjectImgPoints(vxc, vyc, zc, t3d.GetTfromPar(nP_nc), ktt.KI, ktt.K)
    xqr, yqr = t3d.ProjectImgPoints(vxc, vyc, zc*0.8, t3d.GetTfromPar(nP_nc), ktt.KI, ktt.K)
    d = np.sqrt((xpr-xqr)**2+(yqr-ypr)**2+0.1)/2
    dx, dy = (xpr-xqr)/d, (ypr-yqr)/d
    xqr, yqr = xpr-dx, ypr-dy
        
    # blur images (smooth borders) and calculate SAD
    imb = cv2.blur(imn, (6,6))
    cd = np.zeros(np.sum(gd))
    for k, (x0, y0, x1, y1) in enumerate(zip(xpr[gd], ypr[gd], xqr[gd], yqr[gd])):
        w0 = cv2.getRectSubPix(imb, (wsz, wsz), (x0, y0))
        w1 = cv2.getRectSubPix(imb, (wsz, wsz), (x1, y1))
        cd[k] = np.sum(np.abs((w0).astype(np.int)-w1.astype(np.int)))
            
    # return normilized difference (sigmoid used for smoothing and scaling)
    cd = cd/(wsz*wsz)
    cd = 1-turnSig(cd, 8, 4)
    ret_ = np.zeros(len(vxc))
    ret_[gd] = cd
    #t3d.PrintImgPtsDelta(imb, xpr, ypr, xqr, yqr, ret_/800, idx=START+i, outPath='C:/tmp/multiView/im_%04d.png')
    #t3d.PrintImgPtsDelta(imb, xpr[gd], ypr[gd], xqr[gd], yqr[gd], cd, idx=START+i, outPath='C:/tmp/multiView/im_%04d.png')
    return ret_

def SpdDif(x1, y1, x0, y0, spd, nP_nc, scd):
    """speed difference"""
    gd = (((x0-600)**2)/4 + (y0-375)**2) < (150*150)
    s = (get_z_plane(x1[gd], y1[gd], scd) - get_z_plane(x0[gd], y0[gd] - nP_nc[0]*721, scd))
    sd = np.ones(len(x0))*10
    sd[gd] = np.abs(s-spd)
    return sd

def UpdateScaleDelta(nP_nc, scd):
    """scale delta (camera nodding)"""
    if abs(nP_nc[5]) > 0.1:
        dh = (nP_nc[4]/nP_nc[5])*ktt.K[0,0] + ktt.K[1, 2]
        dh = max(165.0,min(185, dh))
        scd = scd*0.75 + dh*0.25
    return scd

def UpdateSpeed(inst, spd, spdVar, w=0.5):
    """smooth update of speed (deprecated)"""
    if abs(spd-inst)<spdVar and inst<0:
        spd = spd*w+inst*(1-w)
        spdVar = max(0.1, spdVar*0.9)
    elif inst<0:
        spdVar = min(3, spdVar*1.3)#+.05)
    return spd, spdVar


def InstSpeed(x0, y0, x1, y1, imc, imn, spd, spdv, nP_nc, scd):
    """# instantaneous speed estimation"""
    #select a single feature based on difference along ep, position and current speed
    d_eig = 0.5*difCrop(x0, y0, x1, y1, imc, imn, nP_nc)
    dist = np.sqrt((y0-385)**2 + ((x0-600)**2) /6)
    d_len = 2*turnSig(dist, 80, 40)
    sd = SpdDif(x1, y1, x0, y0, spd, nP_nc, scd)
    d_spd = 0.5*turnSig(sd, spdv, spdv/2)

    minIdx=np.argmin(d_len+d_eig + d_spd)
    zpl = get_z_plane(x0[minIdx], y0[minIdx], scd)
    m = [minIdx]
    ztr = t3d.GetZ(x0[m], y0[m], x1[m], y1[m], t3d.GetTfromPar(nP_nc), ktt.KI)[0]
    inst = nP_nc[5]*(zpl/ztr)

    # ...
    if i >2:
        ztr2 = t3d.GetZ(xc[m], yc[m], xp[m], yp[m], t3d.Invert(t3d.GetTfromPar(p0)), ktt.KI)[0]
        inst2 = nP_nc[5]*(zpl/ztr2)
        inst = (inst + inst2)/2
    if ((x0[m]-x1[m])**2 +(y0[m]-y1[m])**2)<0.3**2:
        inst = 0
    d = np.min(d_len+d_eig + d_spd)
    
    f = 1
    return inst, d, f


def GetInitSpd(start, init, scd):
    """initial speed estimation"""
    nP_nc = np.zeros(6)
    imp, imc, imn = ktt.GetImg(start), ktt.GetImg(start+1), ktt.GetImg(start+2)
    mrLst, spdLst, pLst = np.zeros(3), np.zeros(3), np.zeros((3,6))
    for spd in [-1, -0.5, -2]:
        nP_nc[5]=spd
        pp, pc, pn = GetPtsTwoDepths(imp, imc, imn, t3d.GetTfromPar(nP_nc), scd)
        (xp, yp), (xc, yc), (xn, yn), idc, i_p = pp.T, pc.T, pn.T, np.arange(len(pp), dtype=np.int), np.arange(len(pp), dtype=np.int)
        p0, nP_nc, mr, cy, fac = BackBa3(idc, i_p, xp, yp, xc, yc, xn, yn, spd=spd, ncyc=max(cyc, 5), rep_thresh=0.25, P_cp=nP_nc, retAll=True, bl=bl)
        inst, dSpd, f = InstSpeed(xc, yc, xn, yn, imc, imn, spd, 3, nP_nc, scd)
        mrLst[i], spdLst[i], pLst[i] = mr, inst, p0
    return spdLst[np.argmin(spdLst)], pLst[np.argmin(spdLst)]


###########################################################################
# POSE ESTIMATION
###########################################################################


def LocateFarClose(gxc, gyc, gzc, gxn, gyn, mr0_lim=0.0, maxTry = 45):
    """use PnP for camera pose location"""
    
    #classify points as far or close
    cls = gzc<np.median(gzc)
    far = np.logical_not(cls)
    mr0 = -1
    for k in range(maxTry):
        # select points randmoly and locate the camera  using PnP
        rind = np.random.randint(0, len(gxc), 10)
        rxc, ryc, rzc, rxn, ryn = gxc[rind], gyc[rind], gzc[rind], gxn[rind], gyn[rind]
        tP_nc, tT_nc = t3d.SimpleLocate(rxc, ryc, rzc, rxn,ryn, ktt.K, ktt.KI)
        
        # calculate the median for close and far points to use as acceptance criterion
        vxp, vyp  = t3d.ProjectImgPoints(gxc, gyc, gzc, tT_nc, ktt.KI, ktt.K)
        mrc = np.median(np.sqrt((vxp[cls]-gxn[cls])**2 + (vyp[cls]-gyn[cls])**2))
        mrf = np.median(np.sqrt((vxp[far]-gxn[far])**2 + (vyp[far]-gyn[far])**2))
        mr = (mrc+mrf)/2
        
        #save the value if it is the best until now
        if (mr < mr0) or mr0<0: 
            mr0, nP_nc, nT_nc = mr, tP_nc, tT_nc
            if mr0 < mr0_lim:
                break
    
    # return the best match
    return mr0, nP_nc, nT_nc

def BackBa3(i_c, i_p, xp, yp, xc, yc, xn, yn, spd=None, ncyc=2, rep_thresh=0.25, P_cp=np.zeros(6), retAll=False, bl=None):
    """cyclic camera pose estimation algorithm:"""

    # invert the last estimation (n-1)->(n) and stop if median projection error below mr0_brake
    T_pc = t3d.Invert(t3d.GetTfromPar(P_cp))
    mr0_brake = 0.15
    for j in range(ncyc):
        if spd is not None:
            T_pc[2, 3] = -1*spd

        # calculate depth using poses (n)->(n-1), and use to calculate pose (n-1)->(n+1)
        vzp = t3d.GetZ(xp, yp, xc, yc, t3d.Invert(T_pc), ktt.KI)
        mr0, P_np, T_np = LocateFarClose(xp, yp, vzp, xn, yn, mr0_lim=0.0, maxTry = RANSAC_TRY)

        # calculate depth using poses (n-1)->(n+1), and use to calculate pose (n+1)->(n)
        vzn = t3d.GetZ(xn, yn, xp, yp, t3d.Invert(T_np), ktt.KI)
        mr1, P_cn, T_cn = LocateFarClose(xn, yn, vzn, xc, yc, mr0_lim=0.0, maxTry = RANSAC_TRY)

        if (j == ncyc-1 or mr1<((j*0.075)+0.125)) and retAll==False:
            break

        # calculate depth using poses (n+1)->(n), and use to calculate pose (n)->(n-1)
        vzc = t3d.GetZ(xc, yc, xn, yn, t3d.Invert(T_cn), ktt.KI)
        mr2, P_pc, T_pc = LocateFarClose(xc, yc, vzc, xp, yp, mr0_lim=0.0, maxTry = RANSAC_TRY)

        mr0_brake +=0.04
        if mr1<((j*0.05)+0.12):
            break

    T_nc = t3d.Invert(T_cn)
    f = 1#FixTz(xp, yp, xc, yc, xn, yn, P_pc, t3d.GetParFromT(T_nc))
    if retAll:
        if spd is not None:
            T_pc[2, 3] = -1*spd
        T_cp = t3d.Invert(T_pc)
        return t3d.GetParFromT(T_cp), t3d.GetParFromT(T_nc), mr1, j, f
    return t3d.GetParFromT(T_nc), mr1, j


def MarkBad(i_c, i_p, xp, yp, xc, yc, xn, yn, p_nc, p_cp, bl, mr):
    """mark as bad features when reprojection error is high"""
    if 1:
        its = np.intersect1d(i_c, i_p)
        pp = np.array([(p in its) for p in i_p], dtype=np.bool)
        pc = np.array([(p in its) for p in i_c], dtype=np.bool)
        xp, yp, xc, yc, xn, yn = xp[pp], yp[pp], xc[pc], yc[pc], xn[pc], yn[pc]
        ic = i_c[pc]

    t0 = t3d.Invert(ktt.poses[START+i+1].reshape(3, 4))
    t1 = ktt.poses[START+i].reshape(3, 4)
    t = t3d.Compose(t0, t1)
    zc = t3d.GetZ(xc, yc, xn, yn, t3d.GetTfromPar(p_nc), ktt.KI)
    xpr, ypr = t3d.ProjectImgPoints(xc, yc, zc, t, ktt.KI, ktt.K)
    d = np.sqrt((xpr-xn)**2+(ypr-yn)**2)
    bad1 = np.zeros(len(xc), dtype=np.bool)

    vzg = get_z_plane(xc, yc, 175)
    bad1 = (yc>220)*(zc>(vzg*1.25))
    if abs(p_nc[5])>1.8:
        bad1 =bad1 +  (yc>220)*(zc<(vzg/1.5))
    if mr<0.5 and p_nc[5]<-1.8 and abs(p_nc[1]) < 0.04:
        da = p_nc[1]*ktt.K[0,0]
        xnn = xn - da
        bad1 = (xc<550)*(xc<(xnn-3))+bad1
        bad1 = (xc>650)*(xc>(xnn+3))+bad1

    if mr>0.4 and False:
        bad1 = bad1 + (d>3)*(zc>70)
        bad1 = bad1 + (d>1)*(zc<3)

    bl = np.hstack((bl, (ic[bad1])))
    bl = np.intersect1d(bl, ic)
    return bl


def fixP(p, p_prev, spd):
    """check for impossible car movement and force a valid value"""
    return p
    if abs(p[1])>0.09 or abs(p[3])>0.4 or abs(p[1] - p_prev[1])>0.03:
        print "fixp", abs(p[1])>0.09, abs(p[3])>0.3, abs(p[1] - p_prev[1])>0.03, "%5.2f, %5.2f"%(abs(p[3]), abs(p[1]))
        p = p_prev
        p[1] = p[1]*0.95
    if p[5]> (-0.001):
        p = np.zeros(6)
    return p


###########################################################################
# main
###########################################################################
# number of cycles and sequence number
for cyc in [3]:
    for seq in [3]:#range(21,-1,-1):

        # initializations
        i=0
        ktt.SetTrackNumber(seq)
        fep, ep = np.zeros((len(ktt.poses), 6)), np.zeros((len(ktt.poses), 6))
        instl, spdl, varl, tml = np.zeros(len(ktt.poses)), np.zeros(len(ktt.poses)), np.zeros(len(ktt.poses)), np.zeros(len(ktt.poses))
        gspd = np.zeros(len(ktt.poses))
        pl = ktt.GetPlst()

        facl, mr_sum, feat_sum = np.zeros(len(ktt.poses)), 0, 0
        START = 0
        #NFRAMES = 200#
        NFRAMES = len(ktt.poses)-1
        bl, scd = np.array([]), 175
        spd, nP_nc, = GetInitSpd(START, 0, scd)
        spdv = 0.3

        xc, yc, xn, yn, idc = np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([]), np.array([]), np.array([])
        xp, yp, i_p = np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        imc, imn = ktt.GetImg(START), ktt.GetImg(START+1)
        t_init, tc = time.time(), time.time()
        for i in range(1, NFRAMES):
            
            # images (previous[n-1], current[n] and next [n+1]); P_nc is the camera  pose 
            # from current->next positions. (xc, yc) are feature coordinates for current image
            imp, imc, imn = imc, imn, ktt.GetImg(START+i+1)
            nP_nc[5]=spd
            pp, pc, pn = GetPtsTwoDepths(imp, imc, imn, t3d.GetTfromPar(nP_nc), scd)
            (xp, yp), (xc, yc), (xn, yn), idc, i_p = pp.T, pc.T, pn.T, np.arange(len(pp), dtype=np.int), np.arange(len(pp), dtype=np.int)

            # after feature tracking perform pose estimation
            if len(xp > 10):
                if i==1:
                    p0, nP_nc, mr, cy, fac = BackBa3(idc, i_p, xp, yp, xc, yc, xn, yn, spd=spd, ncyc=max(cyc, 5), rep_thresh=0.25, P_cp=nP_nc, retAll=True, bl=bl)
                else:
                    p0, nP_nc, mr, cy, fac = BackBa3(idc, i_p, xp, yp, xc, yc, xn, yn, spd=spd, ncyc=cyc, rep_thresh=0.25, P_cp=nP_nc, retAll=True, bl=bl)
                if i>2:
                    nP_nc = fixP(nP_nc, ep[START+i-1], spd)
                    
                scd = UpdateScaleDelta(nP_nc, scd)
                inst, dSpd, f = InstSpeed(xc, yc, xn, yn, imc, imn, spd, spdv, nP_nc, scd)
                
            if spd > -0.01 and inst>-0.01:
                p0, nP_nc = np.zeros(6), np.zeros(6)
            ep[START+i-1] = p0
            ep[START+i] = nP_nc
            if i>1:
                ep[START+i-1] = fixP(p0, ep[START+i-2], spd)

            dspd = spd - nP_nc[5]
            mr_sum, feat_sum = mr_sum+mr, feat_sum+len(xc)
    
            # finally perform scale estimation
            instl[START+i], spdl[START+i], varl[START+i] = inst, spd, spdv
            if abs(dspd)<0.05:
                spd = nP_nc[5]
            elif abs(dspd)<0.1:
                spd = (nP_nc[5]+spd)/2
            facl[i] = fac
            spd, spdv = UpdateSpeed(inst, spd, spdv, w=0.80)

            bl = MarkBad(idc, i_p, xp, yp, xc, yc, xn, yn, ep[START+i], ep[START+i-1], bl, mr)
            avg_time = (time.time() - t_init)/i
            tml[i] = (time.time() - t_init)

            tc = time.time()
            
            if i%200==20:
                print "f:%04d, inst:%4.1f, ty:%5.2f, av_rep:%5.3f, av_t:%5.3f, RT:%d" % (i+START, inst, ep[i, 5], mr_sum/i, avg_time, RANSAC_TRY)# feat_sum/i)

            #if abs(nP_nc[5]) < 0.1:
            #    imc, imn = imp, imc
            #    print ("foo")
            xp, yp, i_p = xc, yc, idc
            xc, yc = xn, yn
    
        if 1:
            np.savetxt("C:/tmp/multiView/frw_pl/p_tr%02d_cyc%02d_wnd%02d_f8.txt" % (ktt.trackNumber, cyc, 3), ep)
        
        tot = time.time() - t_init
        print seq, tot, (tot)/NFRAMES, mr_sum

#plot images:
if 1:
    pl = ktt.GetPlst()
    wpl, wep = ktt.PosesToWorld(pl), ktt.PosesToWorld(ep)

    a=0
    nfig = 7
    fig = plt.figure(nfig); plt.clf()
    
    # plot y angle (car orientation)
    ax0 = fig.add_subplot(221)
    ax0.plot(ep[:, 1], 'k')
    ax0.plot(pl[:, 1], 'g')
    m = min(len(wep), len(wpl))
    ax0.plot(np.unwrap(wep[:m,1] - wpl[:m,1]), 'c')
    
    # plot speed
    ax1 = fig.add_subplot(222)
    ax1.plot(ep[:, 5], 'k')
    ax1.plot(pl[:, 5], 'g')
    
    # plot trajectory
    ax3 = fig.add_subplot(223)
    ax3.plot(wpl[:, 3], wpl[:, 5], 'g', linewidth=2.5)
    ax3.plot(wep[:, 3], wep[:, 5], 'k', linewidth=2.5)
    ax3.axis('equal')
    ax3.grid('on')

    # plot speed (inst and interval)
    ax0 = fig.add_subplot(224)
    ax0.plot(instl, 'mo')
    ax0.plot(ep[:, 5], 'k')
    ax0.plot(pl[:, 5], 'g')
    ax0.plot(spdl + varl, 'c')
    ax0.plot(spdl - varl, 'c')

