#!/bin/bash
# Create a new cid list
truncate cid_list.txt --size 0

# The input filenames
dataName=data1
EOFilePath=../param/$dataName
imgFilePath=../images/$dataName

for file in $EOFilePath/*
do
    imgName=`echo $(basename $file .txt).jpg | sed 's/EO_//g'`
    ./occDetect.py $file $imgFilePath/$imgName
done

# Create a temporary table for id of occluded point set
psql -h localhost -p 5432 -U postgres -d pointdb -c "DROP TABLE IF EXISTS tmpid;CREATE TABLE tmpid (id integer);"

# Copy id of the rows to be removed to temporary table
psql -h localhost -p 5432 -U postgres -d pointdb -c "\COPY tmpid(id) FROM '$(pwd)/cid_list.txt' DELIMITER ' ';"

# Delete rows from colorinfo table and then drop the temporary table
psql -h localhost -p 5432 -U postgres -d pointdb -f sql/removeOcc.sql
rm ./cid_list.txt
