#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distort image coordinates and transform it from mm to pixel."""
import numpy as np
from sympy import lambdify
from sympy import Matrix
from sympy import sqrt
from sympy import symbols


def getIO(IOFileName):
    """Read interior orientation information from file."""
    fin = open(IOFileName)
    data = map(lambda x: float(x), fin.readline().split())
    fin.close()

    # Define interior orientation parameters
    IO = {
        'f': data[0],
        'xp': data[1],
        'yp': data[2],
        'Fw': data[3],
        'Fh': data[4],
        'px': data[5],
        'k1': data[6],
        'k2': data[7],
        'k3': data[8],
        'p1': data[9],
        'p2': data[10]}

    return IO


def xy2RowCol(IO, xcArr, ycArr):
    """Transform given image coordinates from (x, y) to (row, colum)."""
    # Symbols for unknown parameters
    xps, yps = symbols("xp yp")

    # Symbols for constants
    xcs, ycs = symbols("xc yc")

    # Get coordinates of principal point relative to fiducial axis
    x0 = IO['xp'] - IO['Fw']/2
    y0 = -(IO['yp'] - IO['Fh']/2)

    # Compute distance from principal point to image point
    xbar = xps - x0
    ybar = yps - y0
    r = sqrt(xbar**2 + ybar**2)

    # List observation equations
    F = Matrix([
        xcs + x0 - xbar * (r**2*IO['k1']+r**4*IO['k2']+r**6*IO['k3']) -
        (IO['p1']*(r**2+2*xbar**2)+2*IO['p2']*xbar*ybar) - xps,
        ycs + y0 - ybar * (r**2*IO['k1']+r**4*IO['k2']+r**6*IO['k3']) -
        (2*IO['p1']*xbar*ybar+IO['p2']*(r**2+2*ybar**2)) - yps])

    # Create function object for normal matrix N and constant matrix t
    JFx = F.jacobian([xps, yps])
    FuncN = lambdify((xps, yps), JFx.T * JFx)
    Funct = lambdify((xcs, ycs, xps, yps), JFx.T * (-F))

    # Compute corresponding (row, column) values from given (x, y)
    X0 = np.dstack((xcArr, ycArr)).reshape(-1, 2)       # Initial values

    # Iteration process for adding distortion to the corrected x, y
    for i in range(5):
        # Solve unknown parameters
        N = FuncN(X0[:, 0], X0[:, 1])
        f = Funct(xcArr, ycArr, X0[:, 0], X0[:, 1])
        f = np.dstack((f[0], f[1])).reshape(-1, 2, 1)
        dX = np.sum(np.linalg.inv(N.T)*f, axis=1)

        # Update initial values
        X0 = dX + X0

    # Convert from (x, y) to (col, row), and swap these two columns
    tmp = X0[:, 1].copy()
    X0[:, 1] = (X0[:, 0] + IO['Fw']/2.) / IO['px']
    X0[:, 0] = (IO['Fh']/2. - tmp) / IO['px']

    return X0


def main():
    """Perform a simple test for this module."""
    IOFileName = 'input/IO.txt'     # Interior orientation parameter

    # Read interior orientation parameter
    IO = getIO(IOFileName)

    # Get image point coordinates from file
    data = np.genfromtxt("chkRMS/xy2rc/xy.txt", skip_header=1, dtype=[
        ('Name', 'S10'), ('x', 'f8'), ('y', 'f8')])
    x = data['x']
    y = data['y']

    # Compute corrected coordinates
    rowColArr = xy2RowCol(IO, x, y)

    print RowColArr

    return 0

if __name__ == '__main__':
    main()
