#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extracting the corresponding color from image to object points."""
import sys
sys.path.insert(0, "../measureCP")
from pixel2fiducial import allDist
sys.path.insert(0, "../reproject")
from fiducial2pixel import getIO
from fiducial2pixel import xy2RowCol
import hashlib
import numpy as np
from numpy import cos
from numpy import sin
import os
import psycopg2
from scipy.misc import imread
import time


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def getPoint3d(conn, IO, EO):
    """Get the 3d point within the photo scene."""
    cur = conn.cursor()     # Get cursor object of database connection

    # Acquire the average depth of field
    cur.execute("SELECT AVG(X)\nFROM point3d;")
    XA = cur.fetchone()[0]

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
    sql = """
SELECT id, x, y, z
FROM point3d
WHERE Y < %s and Y > %s and Z < %s and Z > %s;"""
    cur.execute(sql, (YA.max(), YA.min(), ZA.max(), ZA.min()))
    objPts = np.array(cur.fetchall(), dtype=[
        ('id', 'i8'), ('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')])

    return objPts


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


def getImgID(conn, imgName):
    """Get the current image ID from database."""
    sql = """
SELECT id FROM image
WHERE position(%s in name) != 0;"""
    cur = conn.cursor()     # Get cursor object of database connection
    cur.execute(sql, (imgName, ))
    imgID = cur.fetchone()[0]

    return imgID


def updateDB(conn, ptSet, EO, imgName):
    """Update the point cloud database."""
    cur = conn.cursor()     # Get cursor object of database connection

    # For image table
    sql = """
INSERT INTO image (id, name, omega, phi, kappa, xl, yl, zl) VALUES
((nextval('image_id_seq'::regclass)), '%s'""" % imgName + ", %.8f" * 6 + ");\n"

    sql = sql % (tuple(EO[3:6]) + tuple(EO[:3]))
    cur.execute(sql)
    conn.commit()

    # Add image id column to ptSet array
    imgIDCol = np.zeros((len(ptSet), 1)) + getImgID(conn, imgName)
    ptSet = np.concatenate((ptSet, imgIDCol), axis=1)

    # Write out the result to temporary file
    h = hashlib.sha1()
    h.update(str(time.time()))
    outputPtFileName = os.path.abspath(h.hexdigest()[:10])

    np.savetxt(
        outputPtFileName,
        ptSet,
        fmt="%d %d %d %.6f %.6f %d %d",
        header="R G B row col point3d_no image_no",
        comments='')

    # For colorinfo table
    sql = """
COPY colorinfo(r, g, b, row, col, point3d_no, image_no)
FROM %s DELIMITER \' \' CSV HEADER;"""
    cur.execute(sql, (outputPtFileName, ))
    conn.commit()

    # Remove temporary file
    os.remove(outputPtFileName)


def main():
    # Define database connection parameters
    host = 'localhost'
    port = '5432'
    dbName = 'pointdb'
    user = 'postgres'

    # Define file names
    IOFileName = '../param/IO.txt'
    EOFileName = '../param/EO_P1_C.txt'
    imgFileName = '../images/P1_C.jpg'

    IO = getIO(IOFileName)

    # XL, YL, ZL, O, P, K, SigXL, SigYL, SigZL, SigO, SigP, SigK
    EO = np.genfromtxt(EOFileName)

    # Connect to database
    try:
        conn = psycopg2.connect(
            "dbname='%s' user='%s' host='%s' port='%s'" %
            (dbName, user, host, port))
    except psycopg2.OperationalError:
        print "Unable to connect to the database."
        return -1

    # Get the target point cloud
    objPts = getPoint3d(conn, IO, EO)

    # Reproject object points to image plane
    print "Reprojecting object points..."
    x, y = getxy(IO, EO, objPts)
    rowColArr = xy2RowCol(IO, x, y)

    # Extract R, G, B color
    print "Resampling color from image..."
    img = imread(imgFileName)
    RGB = extractColor(rowColArr, img)

    # Use the same order as the 'colorinfo' table
    ptSet = np.concatenate((
        RGB, rowColArr, objPts['id'].reshape(-1, 1)), axis=1)

    # Keep the points whose color are not equal to black
    ptSet = ptSet[RGB.sum(axis=1) != -3].view()

    print "Updating point cloud database..."
    updateDB(conn, ptSet, EO, imgFileName)
    conn.close()


if __name__ == '__main__':
    main()
