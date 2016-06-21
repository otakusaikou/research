#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Distort image coordinates and transform it from mm to pixel."""
import numpy as np
from sympy import lambdify
from sympy import Matrix
from sympy import sqrt
from sympy import symbols
import sys


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

    # Create function object for F and its jacobian matrix
    JFx = F.jacobian([xps, yps])
    FuncJFx = lambdify((xcs, ycs, xps, yps), JFx)
    FuncF = lambdify((xcs, ycs, xps, yps), F)

    # Compute corresponding (row, column) values from given (x, y)
    idx = 0
    curValue = 0    # Current percentage of completion
    ptNum = len(xcArr)
    rowColArr = np.zeros((ptNum, 2))
    sys.stdout.write("(x, y) -> (row, col): %3d%%" % 0)

    for xc, yc in zip(xcArr, ycArr):
        X0 = np.matrix([xc, yc]).T      # Initial values for unknown parameters
        dX = np.ones(1)                 # Initial value for iteration

        # Iteration process
        while abs(dX.sum()) > 10**-6:
            # Compute coefficient matrix and constants matrix
            B = FuncJFx(*tuple(np.append((xc, yc), X0)))
            f = -FuncF(*tuple(np.append((xc, yc), X0)))

            # Solve the unknown parameters
            N = B.T * B
            t = B.T * f
            dX = N.I * t

            X0 += dX            # Update initial values

        xp, yp = map(float, X0)

        # From fiducial coordinate system to row and column
        col = (xp + IO['Fw']/2.) / IO['px']
        row = (IO['Fh']/2. - yp) / IO['px']

        rowColArr[idx, :] = row, col
        idx += 1

        # Update the percentage of completion
        if curValue < int(100.0 * idx / ptNum):
            curValue = int(100.0 * idx / ptNum)
            sys.stdout.write("\b" * 4)
            sys.stdout.write("%3d%%" % curValue)
            sys.stdout.flush()
    sys.stdout.write("\n")

    return rowColArr


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

    print rowColArr

    return 0

if __name__ == '__main__':
    main()
