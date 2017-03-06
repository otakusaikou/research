#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../measureCP")
from pixel2fiducial import allDist
sys.path.insert(0, "../reproject")
from fiducial2pixel import getIO
import numpy as np
import os
import pandas as pd
import pcl
from reproject import getM
from scipy.misc import imread
from scipy import spatial


def getParam(refPts, thres):
    """Compute parameters of plane after RANSAC plane fitting process."""
    # Transform to pcl format
    p = pcl.PointCloud(refPts.astype(np.float32))
    p.to_file("._tmp")

    # Remove outlers with RANSAC method (written with C++ code)
    os.popen("./planar_segmentation ._tmp ._tmp %f" % thres).read()

    # Compute the normal vector with inliers
    refPts = pd.read_csv("._tmp", delimiter=' ').values
    mean = np.mean(refPts, axis=0)
    ptSet_adjust = refPts - mean
    matrix = np.cov(ptSet_adjust.T)
    eigVal, eigVec = np.linalg.eigh(matrix)
    a, b, c = eigVec[:, 0]

    # Compute the plane parameter 'd' and sigma0
    d = (a*refPts[:, 0] + b*refPts[:, 1] + c*refPts[:, 2]).mean()
    res = ((a*refPts[:, 0] + b*refPts[:, 1] + c*refPts[:, 2] - d)**2).sum()
    s0 = np.sqrt(res / (refPts.shape[0] - 1))

    # Remove temporary file
    os.remove('._tmp')

    return (a/d, b/d, c/d), (s0,)


def rebuild(IOFileName, EOFileName, outputFileName, ri, ci, params):
    """Rebuild planar surface with given normal vector and image points."""
    # Preprocess of I.O. and E.O. parameters
    IO = getIO(IOFileName)
    f = IO['f']

    EO = np.genfromtxt(EOFileName)
    XL, YL, ZL = EO[:3]
    Omega, Phi, Kappa = map(np.radians, EO[3:6])

    # Transform the image points from pixel to mm
    xc, yc = allDist(ri, ci, IO)

    # Generate new object points
    M = getM(Omega, Phi, Kappa)

    # Reference: https://goo.gl/JFWbDY
    a, b, c = params
    xbar = xc
    ybar = yc
    A = (a*(M[0, 0]*xbar + M[1, 0]*ybar - M[2, 0]*f) +
         b*(M[0, 1]*xbar + M[1, 1]*ybar - M[2, 1]*f) +
         c*(M[0, 2]*xbar + M[1, 2]*ybar - M[2, 2]*f)) / (
             M[0, 2]*xbar + M[1, 2]*ybar - M[2, 2]*f)

    B = 1 - a*XL - b*YL + ZL*(a*(M[0, 0]*xbar + M[1, 0]*ybar - M[2, 0]*f) +
                              b*(M[0, 1]*xbar + M[1, 1]*ybar - M[2, 1]*f)) / (
                                  M[0, 2]*xbar + M[1, 2]*ybar - M[2, 2]*f)

    XA = (M[0, 0]*xbar + M[1, 0]*ybar - M[2, 0]*f) / (
        M[0, 2]*xbar + M[1, 2]*ybar - M[2, 2]*f) * (B/A - ZL) + XL

    YA = (M[0, 1]*xbar + M[1, 1]*ybar - M[2, 1]*f) / (
        M[0, 2]*xbar + M[1, 2]*ybar - M[2, 2]*f) * (B/A - ZL) + YL

    ZA = (B/A)

    newPt = np.dstack((XA, YA, ZA)).reshape(-1, 3)

    return newPt


def main():
    # Define file names
    IOFileName = '../param/IO.txt'
    EOFileName = '../param/EO_P1_L.txt'
    ptFileName = '../ptCloud/P1_L.txt'
    outputFileName = '../ptCloud/P1_L_addPlane.txt'
    referenceImgName = '../images/P1_L_reference.bmp'
    targetImgName = '../images/P1_L_target.bmp'

    # Threshold for RANSAC plane fitting
    thresAcc = 0.03

    # Threshold for number of reference point
    thresNum = 10

    # Load point data
    ptSet = pd.read_csv(ptFileName, delimiter=' ')

    # Build kdtree for query with image point coordinates
    tree = spatial.cKDTree(ptSet[['row', 'col']].values)

    # Read images where reference and target areas are marked manually
    referenceArea = imread(referenceImgName, True)
    targetArea = imread(targetImgName, True)

    # Rebuild planes with every numeric labels in images
    newPt = np.array([])
    for v in np.unique(referenceArea):
        if v == 255:        # Ignore 255
            print "Ignore label: 255"
            continue

        # Get the reference 3D points with image patch
        refImgPts = np.argwhere(referenceArea == v)
        print "Label: %d\nNumber of pixel: %d" % (v, len(refImgPts))

        distance, location = tree.query(
            refImgPts, k=1, distance_upper_bound=0.5)

        # Keep only valid index only
        idx = location[distance != np.inf]

        if len(idx) <= thresNum:
            print "Insufficient reference point, ignore this image patch!"
            continue

        refPts = ptSet[['X', 'Y', 'Z']].values[idx]

        # Calculate the plane parameters
        params, s0 = getParam(refPts, thresAcc)
        print "a=%.6f, b=%.6f, c=%.6f, sigma0=%.6f\n" % (params + s0)

        # Get target image coordinates within target image patch
        tarImgPts = np.argwhere(targetArea == v)
        ri, ci = np.hsplit(tarImgPts, 2)

        newPt = np.append(newPt, rebuild(IOFileName, EOFileName,
                                         outputFileName, ri, ci, params))

    np.savetxt(
        outputFileName,
        newPt.reshape(-1, 3),
        fmt="%.6f %.6f %.6f 255 0 0",
        header="X Y Z R G B",
        comments='')


if __name__ == "__main__":
    main()
