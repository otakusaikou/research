#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module can export point cloud from database with given image name."""
import numpy as np
import psycopg2


def export(conn, imgFileNameList, outputPtFileNameList):
    """Export the point cloud from database in batches."""
    cur = conn.cursor()     # Get cursor object of database connection
    for i in range(len(imgFileNameList)):
        print "Exporting point cloud from '%s'..." % imgFileNameList[i]
        with open('sql/queryByImage.sql') as sql:
            cur.execute(sql.read(), (imgFileNameList[i], ))
        ptSet = np.array(cur.fetchall())

        # Output the result
        np.savetxt(
            outputPtFileNameList[i],
            ptSet,
            fmt="%.6f %.6f %.6f %d %d %d %.6f %.6f",
            header="X Y Z R G B row col",
            comments='')


def main():
    # Define database connection parameters
    host = 'localhost'
    port = '5432'
    dbName = 'pointdb'
    user = 'postgres'

    # Define file names
    imgFileNameList = [
        'P1_L.jpg', 'P1_C.jpg', 'P2_L.jpg', 'P2_C.jpg', 'P3_L.jpg']
    outputPtFileNameList = [
        '../ptCloud/P1_L.txt', '../ptCloud/P1_C.txt', '../ptCloud/P2_L.txt',
        '../ptCloud/P2_C.txt', '../ptCloud/P3_L.txt']

    # Connect to database
    try:
        conn = psycopg2.connect(
            "dbname='%s' user='%s' host='%s' port='%s'" %
            (dbName, user, host, port))
    except psycopg2.OperationalError:
        print "Unable to connect to the database."
        return -1

    # Start batch processing
    export(conn, imgFileNameList, outputPtFileNameList)
    conn.close()


if __name__ == '__main__':
    main()
