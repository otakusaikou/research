#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import psycopg2


def initDB(host, port, user, dbName):
    """Initialize point cloud database."""
    # Replace the old database and create a new one
    print "Initialize point cloud database..."
    cmdStr = "psql -h %s -p %s -U %s -c \"DROP DATABASE IF EXISTS %s;\"" \
        % (host, port, user, dbName)
    os.popen(cmdStr)

    cmdStr = "psql -h %s -p %s -U %s -c \"CREATE DATABASE %s;\"" \
        % (host, port, user, dbName)
    os.popen(cmdStr)

    cmdStr = "psql -h %s -p %s -U %s -d %s -f sql/dbinit.sql" \
        % (host, port, user, dbName)
    os.popen(cmdStr)


def loadPt(conn, ptFileName):
    """Load the 3d point coordinates to the database."""
    # Read object points
    objPts = np.genfromtxt(ptFileName, dtype=[
        ('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')], skip_header=1)
    X, Y, Z = map(lambda k: objPts[k].view(), list("XYZ"))

    # Update the table
    sql = "INSERT INTO point3d (id, x, y, z) VALUES\n"
    for i in range(len(X) - 1):
        sql += "(nextval('point3d_id_seq'::regclass)"
        sql += ", %.8f, %.8f, %.8f),\n" % (X[i], Y[i], Z[i])

    sql += "(nextval('point3d_id_seq'::regclass)"
    sql += ", %.8f, %.8f, %.8f);\n" % (X[-1], Y[-1], Z[-1])

    cur = conn.cursor()     # Get cursor object of database connection

    cur.execute(sql)
    conn.commit()


def main():
    # Define database connection parameters
    host = 'localhost'
    port = '5432'
    dbName = 'pointdb'
    user = 'postgres'

    # Define file names
    ptFileName = '../ptCloud/XYZ_edited.txt'

    # Ask user whether to reinitialize the database
    flag = raw_input("Initialize database? (Y/N) ").lower()
    while flag not in ['yes', 'no', 'y', 'n']:
        flag = raw_input(
            "Invalid selection (You should input 'Y' or 'N') ").lower()

    if flag in ['Y', 'y', 'Yes', 'yes']:
        initDB(host, port, user, dbName)

    # Ask user whether to reinitialize the database
    flag = raw_input("Load the point cloud? (Y/N) ").lower()
    while flag not in ['yes', 'no', 'y', 'n']:
        flag = raw_input(
            "Invalid selection (You should input 'Y' or 'N') ").lower()

    if flag in ['Y', 'y', 'Yes', 'yes']:
        # Connect to database
        try:
            conn = psycopg2.connect(
                "dbname='%s' user='%s' host='%s' port='%s'" %
                (dbName, user, host, port))
        except psycopg2.OperationalError:
            print "Unable to connect to the database."
            return -1
        loadPt(conn, ptFileName)
        conn.close()


if __name__ == '__main__':
    main()
