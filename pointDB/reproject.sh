#!/bin/bash

# The input filenames
dataName=data1
IOFilePath=../param/IO.txt
EOFilePath=../param/$dataName
imgFilePath=../images/$dataName

for file in $EOFilePath/*
do
    imgName=`echo $(basename $file .txt).jpg | sed 's/EO_//g'`
    #./reproject.py ../param/IO.txt $file $imgFilePath/$imgName
    ./reproject.py $IOFilePath $file $imgFilePath/$imgName
done
