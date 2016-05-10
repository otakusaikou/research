#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A simple test for intensity map generation.

This program project 3D points to Y-Z plane, and use the intensity value to
generate a gray image.

"""
import numpy as np
import pandas as pd
import sys
from scipy.misc import imsave
from scipy import spatial


def genIntensityMap(inputFileName, outputFileName, sf):
    """Generate an intensity map with given point file and scale factor."""
    # Read LiDAR points information
    data = pd.read_csv(inputFileName, header=0, delim_whitespace=True)

    X = -data["Y"]  # Use the Y coordinates as image column
    Y = -data["Z"]  # Use the Z coordinates as image row
    I = data["I"]   # The intensity information

    # Get image resolution
    width = int((X.max() - X.min()) * sf)
    height = int((Y.max() - Y.min()) * sf)

    # Generate grid points
    rangeX = np.linspace(X.min(), X.max(), num=width)
    rangeY = np.linspace(Y.min(), Y.max(), num=height)

    # Range tolerance
    threshold = (((X.max() - X.min()) / width)**2 +
                 ((Y.max() - Y.min()) / height)**2)**0.5

    # Perform inverse distance weighted interpolation
    img = np.zeros(height * width)
    idx = 0

    tree = spatial.cKDTree(zip(X, Y))
    for y in rangeY:
        for x in rangeX:
            distance, location = tree.query(
                [x, y], k=4, distance_upper_bound=threshold)
            if (~np.isinf(distance)).sum():
                img[idx] = ((I[location] / distance).sum() /
                            (1.0 / distance).sum())

            idx += 1
        sys.stdout.write("%3d%%" % (100.0 * idx / (height * width)))
        sys.stdout.flush()
        sys.stdout.write("\b" * 4)

    img = img.reshape(height, width)
    imsave(outputFileName, img.astype(np.uint8))


def main():
    sf = 50     # The scale factor of enlargement
    inputFileName = "../ptCloud/XYZRGBI.txt"
    outputFileName = "Intensity.png"

    genIntensityMap(inputFileName, outputFileName, sf)


if __name__ == '__main__':
    main()
