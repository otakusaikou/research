#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Detect the occluded object points."""
import numpy as np
import os
import psycopg2
from scipy.cluster.vq import kmeans2
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

    # XL, YL, ZL, O, P, K, SigXL, SigYL, SigZL, SigO, SigP, SigK
    EO = np.genfromtxt(EOFileName)

    # Get the 3D edge points and build KD-tree with image coordinates
    cid, X, Y, Z, R, G, B, row, col = np.hsplit(ptSet, 9)
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
