#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing import cpu_count
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import psycopg2
from scipy.cluster.vq import kmeans2
from scipy import spatial
import sys
import warnings


def initTable(cur, conn):
    """Create a new table for the merged color values."""
    sql = """
--merged(id) = point3d(id)
DROP TABLE IF EXISTS merged;
CREATE TABLE merged
(
    id integer,
    r integer,
    g integer,
    b integer,
    CONSTRAINT MERGEPK PRIMARY KEY (id)
);
"""

    cur.execute(sql)
    conn.commit()


def getMaxImgNum(cur):
    """Get the max number of color values to a single point."""
    sql = """
SELECT COUNT(id)
FROM colorinfo
GROUP BY point3d_no
ORDER BY COUNT(id) DESC
LIMIT 1;"""

    cur.execute(sql)
    maxNum = int(cur.fetchone()[0])
    return maxNum


def addSngClr(cur, conn):
    """Add points having a single color value to the merged color table."""
    print "Processing points which have a single color value..."
    sql = """
SELECT MAX(r), MAX(g), MAX(b), point3d_no
FROM colorinfo
GROUP BY point3d_no
HAVING COUNT(id) = 1;"""

    cur.execute(sql)
    ptSet = np.array(cur.fetchall())

    # Update the merged result
    with open('_tmpPtSet.txt', 'a') as fout:
        np.savetxt(
            fout,
            ptSet,
            fmt="%d %d %d %d",
            comments='')


def creMulClr(cur, conn, n):
    """Create a map for 3D point id and multiple color id string."""
    sql = """
--Create pointID-imageID table
DROP TABLE IF EXISTS mulclr;
CREATE TABLE mulclr AS
SELECT DISTINCT
    point3d_no,
    array_to_string(ARRAY_AGG(image_no ORDER BY image_no), ' ') idstr
FROM colorinfo
GROUP BY point3d_no
HAVING COUNT(id ORDER BY id) = %s
ORDER BY idstr;

--Get the number of points for different id string
SELECT idstr
FROM mulclr
GROUP BY idstr
ORDER BY count(point3d_no) DESC;"""

    # Return the number of points in each group and its id string
    cur.execute(sql, (n,))
    idStrList = map(lambda e: e[0], cur.fetchall())
    conn.commit()

    return idStrList


def getEOPair(cur, imgID):
    """Get E.O. parameters of the two image pair with the given image IDs."""
    sql = """
SELECT id, xl, yl, zl, omega, phi, kappa
FROM image
WHERE id = %s OR id = %s
ORDER BY id ASC;"""
    cur.execute(sql, imgID)

    EO1, EO2 = map(
        lambda e: e.ravel(), np.vsplit(np.array(cur.fetchall()), 2))

    return EO1, EO2


def mergeClr(cur, conn, idStrList, imgNum):
    """Filter out the outliers and merge color value of inliers."""
    if imgNum == 2:
        numIDSet = len(idStrList)   # Number of image id string set
        for i, idStr in enumerate(idStrList):
            sys.stdout.write("Processing image id: {%s}, (%d/%d)... %3d%%" %
                             (",".join(idStr.split()), (i+1), numIDSet, 0))
            sys.stdout.flush()

            imgID1, imgID2 = map(int, idStr.split())

            # Get E.O. parameters
            EO1, EO2 = getEOPair(cur, (imgID1, imgID2))

            # Get the 3D object points
            sql = """
SELECT P3D.x, P3D.y, P3D.z, P3D.id
FROM point3d P3D JOIN (
    SELECT point3d_no
    FROM mulclr
    WHERE idstr = %s
    ORDER BY point3d_no ASC) M ON P3D.id = M.point3d_no;"""
            cur.execute(sql, (idStr, ))

            # X Y Z pointID
            P3D = np.array(cur.fetchall())
            ptID = P3D[:, -1].reshape(-1, 1)
            XYZ = pd.DataFrame(P3D[:, :3])

            # Get color information from the image pair
            sql = """
SELECT C.r, C.g, C.b, C.row, C.col
FROM colorinfo C JOIN (
    SELECT point3d_no
    FROM mulclr
    WHERE idstr = %s) M ON C.point3d_no = M.point3d_no
WHERE C.image_no = %s
ORDER BY M.point3d_no ASC;"""

            # Split from column 4 ([R, G, B], [row, col])
            cur.execute(sql, (idStr, imgID1))
            RGB1, rowCol1 = np.hsplit(np.array(cur.fetchall()), [3])

            cur.execute(sql, (idStr, imgID2))
            RGB2, rowCol2 = np.hsplit(np.array(cur.fetchall()), [3])

            # Array for the merged color values
            colorArr = np.zeros((len(XYZ), 3))

            # Get the nearest n image points and trace their object points.
            #
            # Then compute the distance from camera to these object points
            # as well as the standard deviation of each n-distance dataset.
            #
            # If the color difference of the object point is greater than
            # the user defined threshold, then discard the color having higher
            # standard deviation of n-distance dataset.
            #
            # Otherwise, use the mean value as the merged color
            tree1 = spatial.cKDTree(rowCol1)
            sys.stdout.write("\b" * 4)  # Update the percentage of completion
            sys.stdout.write("%3d%%" % 50)
            sys.stdout.flush()

            tree2 = spatial.cKDTree(rowCol2)
            sys.stdout.write("\b" * 4)
            sys.stdout.write("%3d%%" % 75)
            sys.stdout.flush()

            # Check the color difference
            thres = 50
            n = 10
            colorDiff = (RGB1 - RGB2)
            mask = (np.sqrt((colorDiff**2).sum(axis=1)) > thres)

            # KNN search with (row, col)
            dis1, loc1 = tree1.query(
                rowCol1[mask], k=n, distance_upper_bound=3)
            dis2, loc2 = tree2.query(
                rowCol2[mask], k=n, distance_upper_bound=3)

            # For color having difference smaller than the threshold
            colorArr[~mask] = (RGB1[~mask] + RGB2[~mask]) / 2

            # Compute standard deviation of the n-distance
            nPtSet1 = XYZ.loc[loc1.ravel()].values
            disEO1 = np.sqrt(
                ((nPtSet1 - EO1[:3])**2).sum(axis=1)).reshape(-1, n)
            disStd1 = np.nanstd(disEO1, axis=1)

            nPtSet2 = XYZ.loc[loc2.ravel()].values
            disEO2 = np.sqrt(
                ((nPtSet2 - EO2[:3])**2).sum(axis=1)).reshape(-1, n)
            disStd2 = np.nanstd(disEO2, axis=1)

            # Index of the color values having significant color difference
            idx = loc1[:, 0]

            colorArr[idx[disStd1 > disStd2]] = [RGB2[idx[disStd1 > disStd2]]]
            colorArr[idx[disStd1 == disStd2]] = [RGB1[idx[disStd1 == disStd2]]]
            colorArr[idx[disStd1 < disStd2]] = [RGB1[idx[disStd1 < disStd2]]]

            ptSet = np.concatenate((colorArr, ptID), axis=1)

            # Update the percentage of completion
            sys.stdout.write("\b" * 4)
            sys.stdout.write("%3d%%" % 100)
            sys.stdout.flush()
            sys.stdout.write("\n")

            # Update the merged result
            with open('_tmpPtSet.txt', 'a') as fout:
                np.savetxt(
                    fout,
                    ptSet,
                    fmt="%d %d %d %d",
                    comments='')

    else:
        sql = """
--Conbine the pointID-imageID table and it corresponding point id
--and color values, also expand the id string to set of image id
SELECT array_agg(C.r) r, array_agg(C.g) g, array_agg(C.b) b, point3d_no
FROM colorinfo C JOIN (
    SELECT point3d_no, unnest(string_to_array(idstr, ' '))::integer image_no
    FROM mulclr) M USING (point3d_no, image_no)
GROUP BY point3d_no;"""

        cur.execute(sql)

        # R G B pointID
        ptIDColorSet = cur.fetchall()

        # R G B pointID
        ptSet = np.zeros((len(ptIDColorSet), 4), dtype=np.int32)
        numPt = len(ptIDColorSet)

        # Setup the progress value of current task
        sys.stdout.write("Processing points having %d color values... %3d%%" %
                         (imgNum, 0))
        sys.stdout.flush()

        curVal = [0]        # Current percentage of completion

        # Prepare for the multiprocessing
        global classifyClr

        def classifyClr(j):
            # Compute and sum up the distance between each color values
            colorArr = np.array(ptIDColorSet[j][:3]).astype(np.double).T
            disSum = spatial.distance.cdist(
                colorArr, colorArr, 'sqeuclidean').sum(axis=1)

            # Filter out the color values which are different to others
            # Ignore the user warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                label = kmeans2(
                    disSum, np.array([disSum.min(), disSum.max()]), 2)[1]

            inlier = np.where(label == 0)

            # Update the percentage of completion
            if curVal[0] < int(100.0 * (j+1) / numPt):
                curVal[0] = int(100.0 * (j+1) / numPt)
                sys.stdout.write("\b" * 4)
                sys.stdout.write("%3d%%" % curVal[0])
            sys.stdout.flush()

            return colorArr[inlier].mean(axis=0).astype(int)

        # Multi-image color filtering with multiprocessing method
        pool = Pool(processes=cpu_count())
        ptSet[:, :3] = pool.map(classifyClr, range(numPt))
        pool.close()

        sys.stdout.write("\b" * 4)
        sys.stdout.write("%3d%%" % 100)
        sys.stdout.write("\n")

        # The point id
        ptSet[:, 3] = np.array(ptIDColorSet)[:, 3]

        # Update the merged result
        with open('_tmpPtSet.txt', 'a') as fout:
            np.savetxt(
                fout,
                ptSet,
                fmt="%d %d %d %d",
                comments='')


def exportTable(cur, outputPtFileName):
    """Export the resulting table."""
    sql = """
COPY (
    SELECT P3D.x, P3D.y, P3D.z, M.r, M.g, M.b
    FROM point3d P3D JOIN merged M USING (id)
)
TO STDOUT DELIMITER \' \' CSV HEADER;"""

    with open(outputPtFileName, 'w') as fout:
        cur.copy_expert(sql, fout)


def main():
    # Define database connection parameters
    host = 'localhost'
    port = '5432'
    dbName = 'pointdb'
    user = 'postgres'

    # Output filename
    outputPtFileName = './output/result.txt'

    # Connect to database
    try:
        conn = psycopg2.connect(
            "dbname='%s' user='%s' host='%s' port='%s'" %
            (dbName, user, host, port))
    except psycopg2.OperationalError:
        print "Unable to connect to the database."
        return -1

    cur = conn.cursor()     # Get cursor object of database connection

    # Ask user whether to reinitialize the table of merged color values
    flag = raw_input("Initialize the merged color table? (Y/N) ").lower()
    while flag not in ['yes', 'no', 'y', 'n']:
        flag = raw_input(
            "Invalid selection (You should input 'Y' or 'N') ").lower()

    if flag in ['Y', 'y', 'Yes', 'yes']:
        initTable(cur, conn)

    # Check the max number of colors
    maxNum = getMaxImgNum(cur)
    print "Max number of colors: %d" % maxNum

    # Create temporary file for the merged color values
    with open('_tmpPtSet.txt', 'w') as fout:
        fout.write("R G B ID\n")

    # Process ponints having a single color value
    addSngClr(cur, conn)

    # Merge the color values
    for i in range(2, maxNum + 1):
        idStrList = creMulClr(cur, conn, i)
        mergeClr(cur, conn, idStrList, i)

    # Update the merged color table
    sql = """
COPY merged(r, g, b, id)
FROM STDIN DELIMITER \' \' CSV HEADER;"""
    with open('_tmpPtSet.txt') as fin:
        cur.copy_expert(sql, fin)
    conn.commit()

    # Remove temporary file and export the final result
    os.remove('_tmpPtSet.txt')
    exportTable(cur, outputPtFileName)

    # Close the connection
    conn.close()


if __name__ == "__main__":
    main()
