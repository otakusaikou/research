#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Detect the occluded object points."""
import numpy as np
import os
import psycopg2
from scipy import spatial
import sys


np.set_printoptions(suppress=True)  # Disable scientific notation for numpy


def occDetect(conn, EOFileName, imgFileName):
    """Seperate normal and occluded points then output them respectively."""
    print "Processing point cloud within image file: " + imgFileName + "..."
    cur = conn.cursor()     # Get cursor object of database connection
    imgFileName = os.path.split(imgFileName)[-1]
    with open('sql/queryByImage.sql') as sql:
        cur.execute(sql.read(), (imgFileName, ))
    ptSet = np.array(cur.fetchall())
    XYZ = ptSet[:, 1:4]

    # XL, YL, ZL, O, P, K, SigXL, SigYL, SigZL, SigO, SigP, SigK
    EO = np.genfromtxt(EOFileName)

    cid = ptSet[:, 0]   # Color id

    # Create the KD-tree
    imgPt = ptSet[:, 7:9]
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

    # Write out the color id of occluded point set
    with open("cid_list.txt", 'a') as fout:
        np.savetxt(
            fout,
            cid[~mask],
            fmt="%d",
            comments='')


def main():
    # Define database connection parameters
    host = 'localhost'
    port = '5432'
    dbName = 'pointdb'
    user = 'postgres'

    # Parse user input if it has any
    # You can use bath script to automate the update process
    # e.g.
    # ----with /bin/bash---- #
    # for file in ../param/EO*
    # do
    #     imgName=${file/param\/EO_/images\/}
    #     ./occDetect.py $file ${imgName%.txt}.jpg
    # done
    #
    if len(sys.argv) != 1:
        EOFileName, imgFileName = sys.argv[1:]
    else:
        # Define file names
        imgFileName = 'IMG_8694.jpg'
        EOFileName = '../param/EO_IMG_8694.txt'

    # Connect to database
    try:
        conn = psycopg2.connect(
            "dbname='%s' user='%s' host='%s' port='%s'" %
            (dbName, user, host, port))
    except psycopg2.OperationalError:
        print "Unable to connect to the database."
        return -1

    occDetect(conn, EOFileName, imgFileName)


if __name__ == "__main__":
    main()
