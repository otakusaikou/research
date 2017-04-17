#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Detect the occluded object points."""
import glob
import numpy as np
import pandas as pd
from scipy import spatial


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def occDetect(EOFileName, ptFileName, cleanedPtFileName, occlusionPtFileName):
    """Seperate normal and occluded points then output them respectively."""
    print "Processing point file: " + ptFileName + "..."
    # XL, YL, ZL, O, P, K, SigXL, SigYL, SigZL, SigO, SigP, SigK
    EO = np.genfromtxt(EOFileName)

    ptSet = pd.read_csv(ptFileName, delimiter=' ')
    XYZ = ptSet[['X', 'Y', 'Z']].values

    # Create the KD-tree
    imgPt = ptSet[['row', 'col']].values
    tree = spatial.cKDTree(imgPt)

    # Get the candidate set of occluded points
    k = 10
    dis, idx = tree.query(imgPt, k=k+1, distance_upper_bound=1.5)

    # Threshold for second step of occlusion filtering
    thresStd = 0.05

    # Number of point which are within the distance upper bound
    num = k + 1 - np.isinf(dis).sum(axis=1)

    # Difference in three component between the camera and object point
    dX, dY, dZ = np.hsplit(XYZ - EO[:3], 3)

    # Mask for the occluded object point
    # The occluded points will be labeled as False
    mask = np.zeros(len(ptSet), np.bool)

    # Filter out the occluded points
    mask[idx[num == 1, 0]] = True     # For point sets having only single point

    # Point sets having more than one points
    for n in range(2, k+2):
        # Index for the current point and its n neighbors
        nPtIdx = idx[num == n, :n].ravel()

        # Distance to the camera
        nDisEO = np.sqrt(
            dX[nPtIdx]**2 + dY[nPtIdx]**2 + dZ[nPtIdx]**2).reshape(-1, n)

        # Points having nearest object point equal to itself
        isShortest = np.argmin(nDisEO, axis=1) == 0
        mask[idx[num == n, 0][isShortest]] = True

        # Point sets having small distance variation
        smallStd = nDisEO[~isShortest].std(axis=1) < thresStd
        mask[idx[num == n, 0][~isShortest][smallStd]] = True

    # Write out the results
    np.savetxt(
        cleanedPtFileName,
        ptSet[mask],
        fmt="%.6f %.6f %.6f %d %d %d %.6f %.6f",
        header="X Y Z R G B row col",
        comments='')

    np.savetxt(
        occlusionPtFileName,
        ptSet[~mask],
        fmt="%.6f %.6f %.6f %d %d %d %.6f %.6f",
        header="X Y Z R G B row col",
        comments='')


def main():
    # Define file names
    EOFileList = glob.glob('../param/data2/EO_IMG_8713*')
    ptFileList = map(
        lambda f: f.replace("param", "ptCloud").replace("EO_", ""), EOFileList)

    cleanedPtFileList = map(
        lambda f: f.replace("ptCloud/", "ptCloud/cleaned/"), ptFileList)

    occlusionPtFileList = map(
        lambda f: f.replace("cleaned", "occlusion"), cleanedPtFileList)

    for i in range(len(EOFileList)):
        occDetect(
            EOFileList[i],
            ptFileList[i],
            cleanedPtFileList[i],
            occlusionPtFileList[i])


if __name__ == "__main__":
    main()
