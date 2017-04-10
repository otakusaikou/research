#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import psycopg2
from scipy.cluster.vq import kmeans2
from scipy.spatial import distance
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
    """Get the max number of color values to single point."""
    sql = """
SELECT COUNT(id)
FROM colorinfo
GROUP BY point3d_no
ORDER BY COUNT(id) DESC
LIMIT 1;"""

    cur.execute(sql)
    maxNum = int(cur.fetchone()[0])
    return maxNum


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
SELECT DISTINCT COUNT(point3d_no), idstr
FROM mulclr
GROUP BY idstr
ORDER BY count(point3d_no) DESC;"""

    # Return the number of points in each group and its id string
    cur.execute(sql, (n,))
    idStrArr = np.array(
        cur.fetchall(), dtype=[('num', 'i8'), ('idstr', 'S50')])
    conn.commit()

    return idStrArr


def mergeClr(cur, conn, idStrArr):
    """Filter out the outliers and merge color value of inliers."""
    for i in range(len(idStrArr)):
        idStr = idStrArr['idstr'][i]

        sql = """
--Conbine the pointID-imageID table and it corresponding point id
--and color values, also expand the id string to set of image id
SELECT array_agg(C.r) r, array_agg(C.g) g, array_agg(C.b) b, point3d_no
FROM colorinfo C JOIN (
    SELECT point3d_no, unnest(string_to_array(idstr, ' '))::integer image_no
    FROM mulclr
    WHERE idstr = %s) M USING (point3d_no, image_no)
GROUP BY point3d_no;"""

        cur.execute(sql, (idStr,))

        # R G B pointID
        ptIDColorSet = cur.fetchall()

        # R G B pointID
        ptSet = np.zeros((len(ptIDColorSet), 4), dtype=np.int32)
        numPt = len(ptIDColorSet)

        # Setup the progress value of current task
        sys.stdout.write("Processing image id: {%s}... %3d%%" %
                         (",".join(idStr.split()), 0))
        curVal = 0          # Current percentage of completion
        for i in range(numPt):
            # Compute and sum up the distance between each color values
            colorArr = np.array(ptIDColorSet[i][:3]).astype(np.double).T
            disSum = distance.cdist(
                colorArr, colorArr, 'sqeuclidean').sum(axis=1)

            # Filter out the color values which are different to other ones
            # Ignore the user warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                label = kmeans2(
                    disSum, np.array([disSum.min(), disSum.max()]), 2)[1]

            inlier = np.where(label == 0)
            ptSet[i, :3] = colorArr[inlier].mean(axis=0).astype(int)

            # The point id
            ptSet[i, 3] = ptIDColorSet[i][3]

            # Update the percentage of completion
            if curVal < int(100.0 * (i+1) / numPt):
                curVal = int(100.0 * (i+1) / numPt)
                sys.stdout.write("\b" * 4)
                sys.stdout.write("%3d%%" % curVal)
                sys.stdout.flush()
        sys.stdout.write("\n")

        np.savetxt(
            '_tmpPtSet.txt',
            ptSet,
            fmt="%d %d %d %d",
            header="",
            comments='')

        # Update the merged result
        sql = """
COPY merged(r, g, b, id)
FROM %s DELIMITER \' \' CSV HEADER;"""
        cur.execute(sql, (os.path.abspath('_tmpPtSet.txt'), ))
        conn.commit()


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
    if maxNum < 3:
        print "The max number of colors is lower than 3, ignore current task."
        return -1
    else:
        print "Max number of colors: %d" % maxNum


    # Merge the color values
    for i in range(3, maxNum + 1):
        idStrArr = creMulClr(cur, conn, i)
        mergeClr(cur, conn, idStrArr)

    # Remove temporary file and export the final result
    os.remove('_tmpPtSet.txt')
    exportTable(cur, outputPtFileName)


if __name__ == "__main__":
    main()
