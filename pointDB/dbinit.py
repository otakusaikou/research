#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os


def initDB(host, port, user, dbName):
    """Initialize point cloud database."""
    # Replace the old database and create a new one
    print "Kill existing session..."
    with open('sql/killsession.sql') as fin:
        sql = fin.read() % dbName
        cmdStr = "psql -h %s -p %s -U %s -c \"%s\"" \
            % (host, port, user, sql)
    os.popen(cmdStr)

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


def main():
    # Define database connection parameters
    host = 'localhost'
    port = '5432'
    dbName = 'pointdb'
    user = 'postgres'

    # Define file names
    ptFileName = os.path.abspath('../ptCloud/XYZ_edited_full_addFull.txt')

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
        cmdStr = ("psql -h %s -p %s -U %s -d %s -c " +
                  "\"\\COPY point3d(X, Y, Z, I) FROM \'%s\' " +
                  "DELIMITER \' \' CSV HEADER;\"") \
            % (host, port, user, dbName, ptFileName)
        os.popen(cmdStr)


if __name__ == '__main__':
    main()
