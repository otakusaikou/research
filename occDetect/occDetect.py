#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Detect the occluded object points."""
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy import spatial


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def occDetect(EOFileName, ptFileName, cleanedPtFileName, occlusionPtFileName):
    """Seperate normal and occluded points then output them respectively."""
    print "Processing point file: " + ptFileName + "..."
    # XL, YL, ZL, O, P, K, SigXL, SigYL, SigZL, SigO, SigP, SigK
    EO = np.genfromtxt(EOFileName)

    # Get the 3D edge points and build KD-tree with image coordinates
    ptSet = pd.read_csv(ptFileName, delimiter=' ').values
    X, Y, Z, R, G, B, row, col = np.hsplit(ptSet, 8)
    tree = spatial.cKDTree(np.dstack((col, row)).reshape(-1, 2))

    # Get the candidate set of occluded points
    k = 8
    dis, idx = tree.query(
        np.dstack((col, row)).reshape(-1, 2),
        k=k+1)

    # Compute distance from exposure station to object points
    disEO = np.hypot(
        X[idx].reshape(-1, k+1) - EO[0],
        Y[idx].reshape(-1, k+1) - EO[1],
        Z[idx].reshape(-1, k+1) - EO[2])

    # First step of occlusion detection. In this step, we classify the
    # candidate point sets with distance to camera. If the standard
    # deviation is relatively big, then label these points as occluded area.
    disStd = disEO.std(axis=1)
    label1 = kmeans2(disStd*10, np.array([0, 5]), 2)[1]     # Use centmeter
    occSet = disEO[label1 == 1]
    occSetIdx = idx[label1 == 1]    # Index of occluded point sets

    # Second step of occlusion detection. We normalize the labeled distances
    # firstly, then classify these point set again. Points with higher index
    # value will be labeled as occluded object points.
    occSetStd = occSet.std(axis=1)
    outSetMin = occSet.min(axis=1)

    # Normalized Index = (D_i - min(D_i) / std(D_i))
    NI = (occSet - outSetMin.reshape(-1, 1)) / occSetStd.reshape(-1, 1)
    label2 = kmeans2(NI.ravel(), np.array([0, 1]), 2)[1]
    idx = np.unique(occSetIdx.ravel()[label2 == 1])
    mask = np.ones(len(ptSet), np.bool)
    mask[idx] = 0

    # Write out the results
    np.savetxt(
        occlusionPtFileName,
        ptSet[~mask],
        fmt="%.6f %.6f %.6f %d %d %d %.6f %.6f",
        header="X Y Z R G B row col",
        comments='')

    np.savetxt(
        cleanedPtFileName,
        ptSet[mask],
        fmt="%.6f %.6f %.6f %d %d %d %.6f %.6f",
        header="X Y Z R G B row col",
        comments='')


def main():
    # Define file names
    EOFileList = [
        '../param/EO_P1_L.txt',
        '../param/EO_P1_C_more.txt',
        '../param/EO_P1_R_more.txt',
        '../param/EO_P2_L_less.txt',
        '../param/EO_P2_C_more.txt',
        '../param/EO_P2_R_more.txt',
        '../param/EO_P3_L_more.txt']

    ptFileList = [
        '../ptCloud/P1_L.txt',
        '../ptCloud/P1_C.txt',
        '../ptCloud/P1_R.txt',
        '../ptCloud/P2_L.txt',
        '../ptCloud/P2_C.txt',
        '../ptCloud/P2_R.txt',
        '../ptCloud/P3_L.txt']

    cleanedPtFileList = [
        '../ptCloud/cleaned/P1_L_clean.txt',
        '../ptCloud/cleaned/P1_C_clean.txt',
        '../ptCloud/cleaned/P1_R_clean.txt',
        '../ptCloud/cleaned/P2_L_clean.txt',
        '../ptCloud/cleaned/P2_C_clean.txt',
        '../ptCloud/cleaned/P2_R_clean.txt',
        '../ptCloud/cleaned/P3_L_clean.txt']

    occlusionPtFileList = [
        '../ptCloud/occlusion/P1_L_occ.txt',
        '../ptCloud/occlusion/P1_C_occ.txt',
        '../ptCloud/occlusion/P1_R_occ.txt',
        '../ptCloud/occlusion/P2_L_occ.txt',
        '../ptCloud/occlusion/P2_C_occ.txt',
        '../ptCloud/occlusion/P2_R_occ.txt',
        '../ptCloud/occlusion/P3_L_occ.txt']

    for i in range(len(EOFileList)):
        occDetect(
            EOFileList[i],
            ptFileList[i],
            cleanedPtFileList[i],
            occlusionPtFileList[i])

if __name__ == "__main__":
    main()
