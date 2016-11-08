#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This is a simple converter.

This program can convert the point cloud with
obj format to ASCII file.
"""
import numpy as np
import os
import pandas as pd
import sys


def main():
    # Process the input/output file names
    inputFileName = sys.argv[-1]
    outputPtFileName = os.path.splitext(inputFileName)[0] + ".txt"

    # Read obj file
    data = pd.read_csv(
        inputFileName,
        delimiter=' ',
        names=['_', 'X', 'Y', 'Z', 'R', 'G', 'B'],
        skiprows=11,
        skipfooter=5,
        engine='python')

    # Transform the RGB value to 8bit unsigned integer
    data.R = (data.R * 255).round()
    data.B = (data.B * 255).round()
    data.G = (data.G * 255).round()

    # Generate point sets and save the result
    ptSet = data[['X', 'Y', 'Z', 'R', 'G', 'B']].values
    np.savetxt(
        outputPtFileName,
        ptSet,
        fmt="%.6f %.6f %.6f %d %d %d",
        header="X Y Z R G B",
        comments='')


if __name__ == '__main__':
    main()
