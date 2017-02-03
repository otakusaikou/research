#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../SIFTMatching")
from SIFTMatching import match
sys.path.insert(0, "../measureCP")
from pixel2fiducial import allDist
from measureCP import getIO
sys.path.insert(0, "../reproject")
from reproject import getxy
from reproject import xy2RowCol
import numpy as np
from scipy.misc import imread
import pandas as pd
from scipy import spatial


def main():
    # Define input file names
    EOFileName = '../param/EO_P1_L.txt'
    IOFileName = '../param/IO.txt'
    ptFileName = '../ptCloud/P1_L.txt'
    imgFileName = '../images/P1_C.jpg'
    outputFileName = 'result.txt'

    # SIFT matching parameters
    ratio = 0.8
    showResult = True

    # Use n nearest neighbor point value for image interpolation
    n = 4

    # Read point cloud information
    data = pd.read_csv(ptFileName, header=0, delim_whitespace=True)

    # Read interior/exterior orientation parameters
    IO = getIO(IOFileName)

    # XL, YL, ZL, O, P, K, SigXL, SigYL, SigZL, SigO, SigP, SigK
    EO = np.genfromtxt(EOFileName)

    # Reproject the object point to reference image plane
    x, y = getxy(IO, EO, data)
    rowColArr = xy2RowCol(IO, x, y)

    # Create KD-tree object with projected point coordinate (col, row)
    tree = spatial.cKDTree(np.roll(rowColArr, 1, axis=1))

    # Generate a false image for matching process between
    # object and image points
    targetImg = imread(imgFileName)
    height, width = targetImg.shape[:2]
    xi, yi = np.meshgrid(np.arange(width), np.arange(height))
    pts = np.dstack((xi.ravel(), yi.ravel())).reshape(-1, 2)

    distance, location = tree.query(
        pts, k=n, distance_upper_bound=1.5)
    mask = np.sum(~np.isinf(distance), axis=1) != 0

    valR = data.R[location[mask].ravel()].values.reshape(-1, n)
    valR[np.isnan(valR)] = 0
    valG = data.G[location[mask].ravel()].values.reshape(-1, n)
    valG[np.isnan(valG)] = 0
    valB = data.B[location[mask].ravel()].values.reshape(-1, n)
    valB[np.isnan(valB)] = 0
    weight = 1.0 / distance[mask]

    falseImg = np.zeros((1280, 1920, 3), dtype=np.uint8)
    falseImg[mask.reshape(xi.shape), 0] = np.sum(valR * weight, axis=1) / \
        np.sum(weight, axis=1)
    falseImg[mask.reshape(xi.shape), 1] = np.sum(valG * weight, axis=1) / \
        np.sum(weight, axis=1)
    falseImg[mask.reshape(xi.shape), 2] = np.sum(valB * weight, axis=1) / \
        np.sum(weight, axis=1)

    # Perform SIFT matching with LiDAR image and photo
    falseImgPt, imgPt = map(lambda x: x.reshape(-1, 2), match(
        falseImg,
        targetImg,
        ratio,
        show=showResult))

    # Query 3D coordinates with nearest false image points
    dis, loc = tree.query(falseImgPt, k=1)
    imgPt = np.dstack((allDist(imgPt[:, 1], imgPt[:, 0], IO))).reshape(-1, 2)
    objPt = data[['X', 'Y', 'Z']].values[loc]

    ptSet = np.concatenate((imgPt, objPt), axis=1)

    # Write out the result
    np.savetxt(
        outputFileName,
        pd.DataFrame(ptSet).drop_duplicates().values,
        fmt="Pt %.6f %.6f %.6f %.6f %.6f" + " 0.005" * 3,
        header=str(IO['f']),
        comments='')


if __name__ == '__main__':
    main()
