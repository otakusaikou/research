#!/bin/bash
for file in ../images/IMG* ../images/P*
do
    imgName=${file##*/}
    ./export.py $imgName ./output/${imgName%.jpg}.txt
done
