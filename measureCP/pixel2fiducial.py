#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Transform image coordinates from pixel to mm with distortion corrected."""
import numpy as np


def allDist(row, col, IO):
    """Get coordinates with both types of distortion corrected."""
    # Transform row and column to fiducial coordinate system
    xp = col * IO["px"] - IO["Fw"]/2
    yp = -(row * IO["px"] - IO["Fh"]/2)

    # Get coordinates of principal point relative to fiducial axis
    x0 = IO["xp"] - IO["Fw"]/2
    y0 = -(IO["yp"] - IO["Fh"]/2)

    # Compute distance from principal point to image point
    xbar = xp - x0
    ybar = yp - y0
    r = np.hypot(xbar, ybar)

    # Corrected coordinates with origin as principal point
    xc = xp - x0 + xbar * (r**2*IO["k1"]+r**4*IO["k2"]+r**6*IO["k3"]) + \
        (IO["p1"]*(r**2+2*xbar**2)+2*IO["p2"]*xbar*ybar)

    yc = yp - y0 + ybar * (r**2*IO["k1"]+r**4*IO["k2"]+r**6*IO["k3"]) + \
        (2*IO["p1"]*xbar*ybar+IO["p2"]*(r**2+2*ybar**2))

    return xc, yc


def main():
    """Perform a simple test for this module."""
    # Read interior orientation information from file
    fin = open("IO.txt")
    data = map(lambda x: float(x), fin.readline().split())
    fin.close()

    # Define interior orientation parameters
    IO = {
        "f": data[0],
        "xp": data[1],
        "yp": data[2],
        "Fw": data[3],
        "Fh": data[4],
        "px": data[5],
        "k1": data[6],
        "k2": data[7],
        "k3": data[8],
        "p1": data[9],
        "p2": data[10]}

    # Get image point coordinates from file
    data = np.genfromtxt("ImgPts.txt", skip_header=1, dtype=[
        ("Name", "S10"), ("ROW", "f8"), ("COL", "f8")])
    row = data["ROW"]
    col = data["COL"]

    # Compute corrected coordinates
    xc, yc = allDist(row, col, IO)

    # Output results
    fout = open("result.txt", "w")
    fout.write(" ".join('{:^15}'.format(s) for s in ['x', 'y']) + "\n")
    for i in range(len(xc)):
        fout.write("%-15.8f %-15.8f\n" % (xc[i], yc[i]))

    return 0

if __name__ == '__main__':
    main()
