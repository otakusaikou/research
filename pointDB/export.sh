#!/bin/bash

# The input filenames and out path
outputPath=./output
imgArr=(`psql -h localhost -p 5432 -U postgres -d pointdb -c "SELECT * FROM image;" | grep "jpg" | awk 'BEGIN {FS="|"}; {print $2}'`)

for file in ${imgArr[@]}
do
#    ./export.py $imgName ./output/${imgName%.jpg}.txt
    ./export.py $file $outputPath/${file%.jpg}.txt
done
