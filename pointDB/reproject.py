#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extracting the corresponding color from image to object points."""
import sys
sys.path.insert(0, "../reproject")
from fiducial2pixel import getIO
from fiducial2pixel import xy2RowCol
import gc
import numpy as np
from numpy import cos
from numpy import sin
from pixel2fiducial import allDist
from pixel2fiducial import img2frame
import psycopg2
from scipy.misc import imread


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
    if ((x < 0) or (x + 1 > img.shape[1] - 1)) or \
            ((y < 0) or (y + 1 > img.shape[0] - 1)):
        return np.zeros((3)) - 1

    x0, x1 = int(x), int(x + 1)
    y0, y1 = int(y), int(y + 1)

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


def getPoint3d(conn, IO, EO):
    """Get the 3d point within the photo scene."""
    cur = conn.cursor()     # Get cursor object of database connection

    # Acquire the average depth of field
    sql = "SELECT AVG(X)\nFROM point3d;"
    cur.execute(sql)
    XA = cur.fetchone()[0]

    # Compute the four corner coordinates of image (col, row)
    width, height = round(IO['Fw'] / IO['px']), round(IO['Fh'] / IO['px'])
    cnrPt = np.array([(0, 0), (width, 0), (width, height), (0, height)])

    # Get point coordinates relative to fiducial axis
    xf, yf = img2frame(cnrPt[:, 0], cnrPt[:, 1], IO)

    # Compute corrected coordinates
    xc, yc = allDist(xf, yf, IO)

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
SELECT *
FROM point3d
WHERE Y < %s and Y > %s and Z < %s and Z > %s;"""
    cur.execute(sql, (YA.max(), YA.min(), ZA.max(), ZA.min()))
    objPts = np.array(cur.fetchall(), dtype=[
        ('id', 'i8'), ('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')])

    # Free the memory
    del sql, cur
    gc.collect()

    return objPts


def updateDB(conn, ptSet, EO, imgName):
    """Update the point cloud database."""
    cur = conn.cursor()     # Get cursor object of database connection
    numPt = len(ptSet)      # Number of point

    # For image table
    sql = """
INSERT INTO image (id, name, omega, phi, kappa, xl, yl, zl) VALUES
((nextval('image_id_seq'::regclass)), '%s'""" % imgName + ", %.8f" * 6 + ");\n"

    sql = sql % (tuple(EO[3:6]) + tuple(EO[:3]))

    # For point2d table
    sql += "INSERT INTO point2d (id, row, col, image_no) VALUES\n"
    for i in range(numPt - 1):
        sql += "((nextval('point2d_id_seq'::regclass)), "
        sql += "%.8f, %.8f, " % tuple(ptSet[i, -2:])
        sql += "(currval('image_id_seq'::regclass))),\n"

    sql += "((nextval('point2d_id_seq'::regclass)), "
    sql += "%.8f, %.8f, " % tuple(ptSet[-1, -2:])
    sql += "(currval('image_id_seq'::regclass)));\n"

    # For color table
    sql += "INSERT INTO color (id, r, g, b, point3d_no, point2d_no) VALUES\n"
    for i in range(0, numPt - 1):
        sql += "((nextval('color_id_seq'::regclass)), %d, %d, %d, %d, " % \
            (tuple(ptSet[i, 4:7]) + tuple([ptSet[i, 0]]))
        sql += "(currval('point2d_id_seq'::regclass) - %d)),\n" % \
            (numPt - (i + 1))

    sql += "((nextval('color_id_seq'::regclass)), %d, %d, %d, %d, " % \
        (tuple(ptSet[-1, 4:7]) + tuple([ptSet[-1, 0]]))
    sql += "(currval('point2d_id_seq'::regclass)));\n"

    cur.execute(sql)
    conn.commit()

    # Free the memory
    del sql, cur
    gc.collect()


def main():
    # Define database connection parameters
    host = 'localhost'
    port = '5432'
    dbName = 'pointdb'
    user = 'postgres'

    # Define file names
    IOFileName = '../param/IO.txt'
    EOFileName = '../param/EO_P1_L.txt'
    imgFileName = '../images/P1_L.jpg'

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
    x, y = getxy(IO, EO, objPts)
    rowColArr = xy2RowCol(IO, x, y)

    # Extract R, G, B color
    img = imread(imgFileName)
    RGB = extractColor(rowColArr, img)
    ptSet = np.concatenate((
        objPts['id'].reshape(-1, 1),
        objPts[['X', 'Y', 'Z']].view(np.double).reshape(-1, 3),
        RGB, rowColArr), axis=1)

    # Keep the points whose color are not equal to black
    ptSet = ptSet[RGB.sum(axis=1) != -3].view()

    # Free the memory
    del objPts, x, y, img, RGB, rowColArr
    gc.collect()

    updateDB(conn, ptSet, EO, imgFileName)
    conn.close()
    np.savetxt("output.txt", ptSet, fmt="%d, %f, %f, %f, %d, %d, %d, %f, %f")


if __name__ == '__main__':
    main()
