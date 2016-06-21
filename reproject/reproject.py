#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extracting the corresponding color from image to object points."""
from fiducial2pixel import getIO
from fiducial2pixel import xy2RowCol
import numpy as np
from numpy import cos
from numpy import sin
from scipy.misc import imread
import sys


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def getM(Omega, Phi, Kappa):
    """Compute rotation matrix M."""
    M = np.matrix([
        [
            cos(Phi)*cos(Kappa),
            sin(Omega)*sin(Phi)*cos(Kappa) + cos(Omega)*sin(Kappa),
            -cos(Omega)*sin(Phi)*cos(Kappa) + sin(Omega)*sin(Kappa)],
        [
            -cos(Phi)*sin(Kappa),
            -sin(Omega)*sin(Phi)*sin(Kappa) + cos(Omega)*cos(Kappa),
            cos(Omega)*sin(Phi)*sin(Kappa) + sin(Omega)*cos(Kappa)],
        [
            sin(Phi),
            -sin(Omega)*cos(Phi),
            cos(Omega)*cos(Phi)]
        ])

    return M


def getxy(IO, EO, objPts):
    """List observation equations."""
    f, xo, yo = IO['f'], 0, 0     # IO
    XL, YL, ZL = EO[:3]         # EO
    Omega, Phi, Kappa = map(np.radians, EO[3:6])
    XA, YA, ZA = objPts['X'], objPts['Y'], objPts['Z']      # Object points

    M = getM(Omega, Phi, Kappa)

    r = M[0, 0] * (XA - XL) + M[0, 1] * (YA - YL) + M[0, 2] * (ZA - ZL)
    s = M[1, 0] * (XA - XL) + M[1, 1] * (YA - YL) + M[1, 2] * (ZA - ZL)
    q = M[2, 0] * (XA - XL) + M[2, 1] * (YA - YL) + M[2, 2] * (ZA - ZL)

    x = xo - f * (r / q)
    y = yo - f * (s / q)

    return x, y


def getInterpolation(img, x, y):
    """Resample from input image, using bilinear interpolation."""
    # Get coordinates of nearest four points as well as ensuring the
    # coordinates of four points are in the right image extent
    x0, x1 = np.clip([x, x + 1], 0, img.shape[1] - 1).astype(int)
    y0, y1 = np.clip([y, y + 1], 0, img.shape[0] - 1).astype(int)

    # Get intensity of nearest four points
    Ia = img[y0, x0]  # Upper left corner
    Ib = img[y1, x0]  # Lower left corner
    Ic = img[y0, x1]  # Upper right corner
    Id = img[y1, x1]  # Lower right corner

    # Compute the weight of four points
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id


def extractColor(rowColArr, img):
    """Extract color from image."""
    idx = 0
    curValue = 0    # Current percentage of completion
    ptNum = len(rowColArr)
    rgbArr = np.zeros((ptNum, 3))
    sys.stdout.write("Color interpolation process: %3d%%" % 0)

    for row in rowColArr:
        rgbArr[idx, :] = getInterpolation(img, row[1], row[0])
        idx += 1

        # Update the percentage of completion
        if curValue < int(100.0 * idx / ptNum):
            curValue = int(100.0 * idx / ptNum)
            sys.stdout.write("\b" * 4)
            sys.stdout.write("%3d%%" % curValue)
            sys.stdout.flush()
    sys.stdout.write("\n")

    return rgbArr.astype(int)


def main():
    # Define file names
    IOFileName = 'IO.txt'
    EOFileName = 'EO.txt'
    ptFileName = '../ptCloud/objPts.txt'
    imgFileName = '../images/P1_L.jpg'
    outputPtFileName = 'result.txt'

    IO = getIO(IOFileName)

    # XL, YL, ZL, O, P, K, SigXL, SigYL, SigZL, SigO, SigP, SigK
    EO = np.genfromtxt(EOFileName)

    # Read object points
    objPts = np.genfromtxt(ptFileName, dtype=[
        ('Name', 'S15'), ('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')], skip_header=1)

    # Reproject object points to image plane
    x, y = getxy(IO, EO, objPts)
    rowColArr = xy2RowCol(IO, x, y)

    # Extract R, G, B color
    img = imread(imgFileName)
    RGB = extractColor(rowColArr, img)
    ptSet = np.concatenate(
        (objPts[['X', 'Y', 'Z']].view(np.double).reshape(-1, 3), RGB), axis=1)

    # Write out the result
    np.savetxt(
        outputPtFileName,
        ptSet,
        fmt="%.6f %.6f %.6f %d %d %d",
        header="X Y Z R G B",
        comments='')


if __name__ == '__main__':
    main()
