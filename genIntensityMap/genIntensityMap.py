#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A simple test for intensity map generation.

This program project 3D points to Y-Z plane, and use the intensity value to
generate a gray image.

"""
import numpy as np
import pandas as pd
from scipy.misc import imsave
from scipy import spatial


def genIntensityMap(inputFileName, outputFileName, sf, n=4):
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

    # Generate grid points
    xi, yi = np.meshgrid(rangeX, rangeY)
    pts = np.dstack((xi.ravel(), yi.ravel())).reshape(-1, 2)

    # Perform inverse distance weighted interpolation
    tree = spatial.cKDTree(zip(X, Y))
    distance, location = tree.query(
        pts, k=n, distance_upper_bound=threshold)
    mask = np.sum(~np.isinf(distance), axis=1) != 0

    values = I[location[mask].ravel()].reshape(-1, n)
    values[np.isnan(values)] = 0
    weight = 1.0 / distance[mask]

    img = np.zeros(xi.shape)
    img[mask.reshape(xi.shape)] = np.sum(values * weight, axis=1) / \
        np.sum(weight, axis=1)
    imsave(outputFileName, img.astype(np.uint8))


def main():
    sf = 100    # The scale factor of enlargement
    n = 4       # The number of nearest neighbors for kd-tree query
    inputFileName = "../ptCloud/XYZRGBI.txt"
    outputFileName = "Intensity.png"

    genIntensityMap(inputFileName, outputFileName, sf, n)


if __name__ == '__main__':
    main()
