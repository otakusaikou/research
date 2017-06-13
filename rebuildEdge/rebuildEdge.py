#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../imgAnalysis")
from houghTrans import getStrtLine
from imgAnalysis import getLine
import numpy as np
import pandas as pd
from scipy import spatial


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def getEndPt(X, Y, Z, row, col):
    """Get the start and end point of the edge."""
    # For the start point
    idxMin = np.argmin(X + Y + Z)
    Xs, Ys, Zs = X[idxMin, 0], Y[idxMin, 0], Z[idxMin, 0]
    xs, ys = col[idxMin, 0], row[idxMin, 0]

    # For the end point
    idxMax = np.argmax(X + Y + Z)
    Xe, Ye, Ze = X[idxMax, 0], Y[idxMax, 0], Z[idxMax, 0]
    xe, ye = col[idxMax, 0], row[idxMax, 0]

    return (Xs, Ys, Zs, xs, ys), (Xe, Ye, Ze, xe, ye)


def getEdgeParam(Xs, Ys, Zs, X, Y, Z):
    """Estimate the edge parameters."""
    # Estimate the vector of edge
    B = np.matrix(np.zeros((X.shape[0] * 3, 3)))
    f = np.matrix(np.zeros((X.shape[0] * 3, 1)))
    for i in range(X.shape[0]):
        t = np.sqrt((X[i] - Xs)**2 + (Y[i] - Ys)**2 + (Z[i] - Zs)**2)
        B[3 * i: 3 * (i + 1)] = np.identity(3) * t
        f[3 * i] = X[i] - Xs
        f[3 * i + 1] = Y[i] - Ys
        f[3 * i + 2] = Z[i] - Zs

    N = B.T * B         # Compute normal matrix
    t = B.T * f         # Compute t matrix
    dX = N.I * t        # Compute the unknown parameters

    a, b, c = np.array(dX).flatten()

    # Perform error assessment
    V = (f - B * dX)    # Compute residual vector

    # Compute sigma0
    s0 = ((V.T * V)[0, 0] / (B.shape[0] - B.shape[1]))**0.5

    # Compute error of unknown parameters
    # SigmaXX = s0**2 * N.I

    return a, b, c, s0


def main():
    # Define file names
    imgFileName = '../images/data1/P1_R.jpg'
    ptFileName = '../pointDB/addLineTest/P1_R.txt'
    oldLineFileName = './addLine/oldLines_P1R.txt'
    newLineFileName = './addLine/newLines_P1R.txt'

    # Define parameters for canny edge detector and hough transformation
    cannySigma = 3          # Standard deviation of the Gaussian filter
    cannyLowThres = 20      # Lower bound for hysteresis thresholding
    cannyHighThres = 30     # Upper bound for hysteresis thresholding
    houghThres = 25         # Minimum number of intersections to detect a line
    houghLineLength = 30    # Minimum accepted length of detected lines
    houghLineGap = 1        # Maximum gap between pixels to still form a line

    # Threshold for adjustment of edge points
    edgeThres = 0.003

    # Output the parameter settings
    print "cannySigma: %d" % cannySigma
    print "cannyLowThres: %d" % cannyLowThres
    print "cannyHighThres: %d" % cannyHighThres
    print "houghThres: %d" % houghThres
    print "houghLineLength: %d" % houghLineLength
    print "houghLineGap: %d" % houghLineGap
    print "edgeThres: %f" % edgeThres

    # Range tolerance
    threshold = 5

    # Detect the straight lines on the image
    lines = getStrtLine(
        imgFileName,
        (cannySigma, cannyLowThres, cannyHighThres),
        (houghThres, houghLineLength, houghLineGap), True)

    # Get the 3D edge points and build KD-tree with image coordinates
    ptSet = pd.read_csv(ptFileName, delimiter=' ').values
    X, Y, Z, R, G, B, row, col = np.hsplit(ptSet, 8)
    tree = spatial.cKDTree(np.dstack((col, row)).reshape(-1, 2))

    # Rebuild the broken edge
    oldLineFile = open(oldLineFileName, 'w')
    newLineFile = open(newLineFileName, 'w')
    oldLineFile.write("X Y Z R G B\n")
    newLineFile.write("X Y Z R G B\n")

    lineCount = 0
    densRateList = []
    print "Ratio\tBefore\tAfter"
    for line in lines:
        # Generate point coordinates with start and end point
        x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
        xi, yi = getLine((x1, y1), (x2, y2))
        pts = np.dstack((xi.ravel(), yi.ravel())).reshape(-1, 2)

        # Find the nearest projected object point
        idx = -np.ones(len(xi), dtype=int)
        distance, location = tree.query(
            pts, k=1, distance_upper_bound=threshold)

        # Remove invalid point
        location = location[distance != np.inf]
        distance = distance[distance != np.inf]

        # Use only unique point
        _, idx = np.unique(location, return_index=True)
        idx.sort()
        location = location[idx]
        distance = distance[idx]

        # Record valid lines
        for i in range(len(location)):
            oldLineFile.write(
                "%.6f %.6f %.6f %d %d %d\n" % tuple(ptSet[location[i], :-2]))

        # Generate new points
        if len(location) > 10:
            # Get the start and end point coordinates
            startPt, endPt = getEndPt(
                X[location], Y[location], Z[location],
                row[location], col[location])

            Xs, Ys, Zs, xs, ys = startPt
            Xe, Ye, Ze, xe, ye = endPt

            # Estimate the edge parameters
            a, b, c, s0 = getEdgeParam(
                Xs, Ys, Zs, X[location], Y[location], Z[location])

            # Generate the sign parameters
            sign = np.ones((len(xi), 1))
            mainDirect = np.arctan2(ye-ys, xe-xs) % (2*np.pi)
            ptDirect = np.arctan2(yi-ys, xi-xs) % (2*np.pi)
            angleDiff = np.abs(mainDirect - ptDirect)

            sign[(angleDiff > (np.pi/2)) & (angleDiff < (3*np.pi/2))] = -1

            if s0 < edgeThres:
                # Rebuild the object point with image coordinates
                scale = np.sqrt(
                    ((Xe - Xs)**2 + (Ye - Ys)**2 + (Ze - Zs)**2) /
                    ((xe - xs)**2 + (ye - ys)**2))

                t = np.sqrt((xi - xs)**2 + (yi - ys)**2) * scale

                XA = Xs + a * t * sign
                YA = Ys + b * t * sign
                ZA = Zs + c * t * sign

                newPtSet = np.dstack((XA, YA, ZA)).reshape(-1, 3)
                for newPt in newPtSet:
                    newLineFile.write(
                        "%.6f %.6f %.6f 255 0 0\n" % tuple(newPt))

                # Update counters
                lineCount += 1
                densRateList.append(100.0*len(location)/len(xi))

                # Output information on the densification process
                print "%.2f%%\t%d\t%d" % \
                    (100.0*len(location)/len(xi), len(location), len(xi))

    oldLineFile.close()
    newLineFile.close()
    print "Totally %d lines were detected" % len(lines)
    print "Average rate of densification: %.2f%%" % \
        np.array(densRateList).mean()
    print "Success rate: %.2f%%" % (100.0 * lineCount / len(lines))


if __name__ == '__main__':
    main()
