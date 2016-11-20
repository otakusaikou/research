#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extracting the corresponding color from image to object points."""
import sys
sys.path.insert(0, "../measureCP")
from pixel2fiducial import allDist
from fiducial2pixel import getIO
from fiducial2pixel import xy2RowCol
import numpy as np
from numpy import cos
from numpy import sin
from scipy.misc import imread


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def getPoint3d(ptFileName, IO, EO):
    """Get the 3d point within the photo scene."""
    objPts = np.genfromtxt(ptFileName, dtype=[
        ('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')], skip_header=1)

    # Acquire the average depth of field
    XA = objPts['X'].mean()

    # Compute the four corner coordinates of image (col, row)
    width, height = round(IO['Fw'] / IO['px']), round(IO['Fh'] / IO['px'])
    cnrPt = np.array([(0, 0), (width, 0), (width, height), (0, height)])

    # Compute corrected coordinates under the fiducial axis coordinate system
    xc, yc = allDist(cnrPt[:, 1], cnrPt[:, 0], IO)

    f = IO['f']
    XL, YL, ZL = EO[:3]
    Omega, Phi, Kappa = map(np.radians, EO[3:6])
    M = getM(Omega, Phi, Kappa)

    # Compute the image extent in the object space
    a = -(xc * M[2, 0] + f * M[0, 0]) / (xc * M[2, 1] + f * M[0, 1])
    b = -(xc * M[2, 2] + f * M[0, 2]) / (xc * M[2, 1] + f * M[0, 1])
    c = -(yc * M[2, 0] + f * M[1, 0]) / (yc * M[2, 2] + f * M[1, 2])
    d = -(yc * M[2, 1] + f * M[1, 1]) / (yc * M[2, 2] + f * M[1, 2])
    YA = ((a + b * c) / (1 - b * d)) * (XA - XL) + YL
    ZA = ((c + a * d) / (1 - b * d)) * (XA - XL) + ZL

    # Acquire the 3d point within the photo scene
    Y = objPts['Y']
    Z = objPts['Z']
    mask = ((Y < YA.max()) & (Y > YA.min()) & (Z < ZA.max()) & (Z > ZA.min()))

    return objPts[mask]


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


def extractColor(rowColArr, img):
    """Resample from input image, using bilinear interpolation."""
    print "Resampling color from image..."
    ptNum = len(rowColArr)
    rgbArr = np.zeros((ptNum, 3))

    # Get coordinates of nearest four points as well as ensuring the
    # coordinates of four points are in the image extent
    y, x = np.hsplit(rowColArr, 2)
    mask = np.any([(x < 0), (x + 1 > img.shape[1] - 1),
                   (y < 0), (y + 1 > img.shape[0] - 1)], axis=0).ravel()
    rgbArr[mask, :] -= 1        # Mark the outside points as -1

    # Four corner points
    x0 = x[~mask].astype(int)
    x1 = x0 + 1
    y0 = y[~mask].astype(int)
    y1 = y0 + 1

    x = x[~mask]
    y = y[~mask]

    # Get intensity of nearest four points
    Ia = img[y0.ravel(), x0.ravel()]  # Upper left corner
    Ib = img[y1.ravel(), x0.ravel()]  # Lower left corner
    Ic = img[y0.ravel(), x1.ravel()]  # Upper right corner
    Id = img[y1.ravel(), x1.ravel()]  # Lower right corner

    # Compute the weight of four points
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    rgbArr[~mask, :] = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return rgbArr


def main():
    # Define file names
    IOFileName = '../param/IO.txt'
    EOFileName = '../param/EO_P1_L.txt'
    ptFileName = '../ptCloud/XYZ_edited.txt'
    imgFileName = '../images/P1_L.jpg'
    outputPtFileName = '../ptCloud/P1_L.txt'

    IO = getIO(IOFileName)

    # XL, YL, ZL, O, P, K, SigXL, SigYL, SigZL, SigO, SigP, SigK
    EO = np.genfromtxt(EOFileName)

    # Read object points
    objPts = getPoint3d(ptFileName, IO, EO)

    # Reproject object points to image plane
    x, y = getxy(IO, EO, objPts)
    rowColArr = xy2RowCol(IO, x, y)

    # Extract R, G, B color
    img = imread(imgFileName)
    RGB = extractColor(rowColArr, img)
    ptSet = np.concatenate(
        (objPts[['X', 'Y', 'Z']].view(np.double).reshape(-1, 3), RGB), axis=1)

    # Keep the points whose color are not equal to black
    ptSet = ptSet[RGB.sum(axis=1) != -3].view()

    # Write out the result
    np.savetxt(
        outputPtFileName,
        ptSet,
        fmt="%.6f %.6f %.6f %d %d %d",
        header="X Y Z R G B",
        comments='')


if __name__ == '__main__':
    main()
