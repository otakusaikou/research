#!/bin/bash
for file in ../param/EO*
do
    imgName=${file/param\/EO_/images\/}
    ./reproject.py ../param/IO.txt $file ${imgName%.txt}.jpg
done
