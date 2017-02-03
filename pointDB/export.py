#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module can export point cloud from database with given image name."""
import numpy as np
import psycopg2
import sys


def export(conn, imgFileName, outputPtFileName):
    """Export the point cloud from database in batches."""
    cur = conn.cursor()     # Get cursor object of database connection
    print "Exporting point cloud from '%s'..." % imgFileName
    with open('sql/queryByImage.sql') as sql:
        cur.execute(sql.read(), (imgFileName, ))
    ptSet = np.array(cur.fetchall())

    # Check if the data is empty array
    if not ptSet.size:
        print "Cannot found any data from image '%s'" % imgFileName
        return

    # Output the result
    np.savetxt(
        outputPtFileName,
        ptSet[:, 1:],
        fmt="%.6f %.6f %.6f %d %d %d %.6f %.6f",
        header="X Y Z R G B row col",
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
    # for file in ../images/IMG* ../images/P*
    # do
    #     imgName=${file##*/}
    #     echo ./export.py $imgName ./${imgName%.jpg}.txt
    # done
    #
    if len(sys.argv) != 1:
        imgFileName, outputPtFileName = sys.argv[1:]
    else:
        # Define file names
        imgFileName = 'IMG_8694.jpg'
        outputPtFileName = './IMG_8694.txt'

    # Connect to database
    try:
        conn = psycopg2.connect(
            "dbname='%s' user='%s' host='%s' port='%s'" %
            (dbName, user, host, port))
    except psycopg2.OperationalError:
        print "Unable to connect to the database."
        return -1

    export(conn, imgFileName, outputPtFileName)
    conn.close()


if __name__ == '__main__':
    main()
