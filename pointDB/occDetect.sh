#!/bin/bash
# Create a new cid list
truncate cid_list.txt --size 0

for file in ../param/EO*
do
    imgName=${file/param\/EO_/images\/}
    ./occDetect.py $file ${imgName%.txt}.jpg
done

# Create a temporary table for id of occluded point set
psql -h localhost -p 5432 -U postgres -d pointdb -c "DROP TABLE IF EXISTS tmpid;CREATE TABLE tmpid (id integer);"

# Copy id of the rows to be removed to temporary table
psql -h localhost -p 5432 -U postgres -d pointdb -c "COPY tmpid(id) FROM '$(pwd)/cid_list.txt' DELIMITER ' ';"

# Delete rows from colorinfo table and then drop the temporary table
psql -h localhost -p 5432 -U postgres -d pointdb -c "DELETE FROM colorinfo C WHERE C.id IN (SELECT DISTINCT id FROM tmpid);DROP TABLE IF EXISTS tmpid;"
rm ./cid_list.txt
