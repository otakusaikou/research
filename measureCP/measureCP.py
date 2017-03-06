#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from pixel2fiducial import allDist
from scipy.misc import imread
from scipy import spatial


def measureProc(
        intensityImg, rgbImg, LiDARFileName, outputFileName, sf, IOFileName):
    """A function to control measuring process."""
    # Create figure
    fig = plt.figure("Point Measuring Process", figsize=(15, 6))
    ax1 = fig.add_subplot(121)  # For intensity image
    ax1.axis([0, intensityImg.shape[1], intensityImg.shape[0], 0])
    ax1.axis('off')
    ax1.imshow(intensityImg, cmap='Greys_r', interpolation="none")

    ax2 = fig.add_subplot(122)  # For colored photo
    ax2.axis([0, rgbImg.shape[1], rgbImg.shape[0], 0])
    ax2.axis('off')
    ax2.imshow(rgbImg, interpolation="none")

    # Read LiDAR points information
    data = np.genfromtxt(
        LiDARFileName,
        dtype=[('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')],
        skip_header=1,
        usecols=(0, 1, 2))

    # Read interior orientation parameter
    IO = getIO(IOFileName)

    X = -data["Y"]  # Use the Y coordinates as image column
    Y = -data["Z"]  # Use the Z coordinates as image row

    # Get image resolution
    width = int((X.max() - X.min()) * sf)
    height = int((Y.max() - Y.min()) * sf)

    # Generate grid points
    rangeX = np.linspace(X.min(), X.max(), num=width)
    rangeY = np.linspace(Y.min(), Y.max(), num=height)

    # Create KD tree for point query
    tree = spatial.cKDTree(zip(X, Y))

    # Listen mouse button press event
    fig.canvas.mpl_connect('button_press_event', lambda event:
                           mousePress(event, tree, rangeX, rangeY, data, IO))

    # Listen key press event
    fig.canvas.mpl_connect('key_press_event', keyPress)

    plt.tight_layout()
    plt.show()

    # Save the result
    with open(outputFileName, 'w') as fout:
        fout.write("%.6f\n" % IO['f'])
        for pt in ptList:
            fout.write("Point%d" % (ptList.index(pt) + 1))
            fout.write((" %.6f" * 5) % tuple(pt[2:]) + "\n")


def mousePress(event, tree, rangeX, rangeY, data, IO):
    """Handle the mouse press event."""
    if event.inaxes:    # Check if the mouse is in the image area
        ax = event.inaxes
        if event.button == 1 and event.key == 'control':    # Draw point
            # For intensity map
            if ax.colNum == 0:
                distance, location = tree.query(
                    [rangeX[int(event.xdata)], rangeY[int(event.ydata)]],
                    k=1, distance_upper_bound=0.01)

                # Create new point or replace old point
                if location < len(data):
                    ax.plot(event.xdata, event.ydata, "ro")
                    resetPt(ax, (data['X'][location], data['Y'][location],
                            data['Z'][location]))

            # For colored photo
            elif ax.colNum == 1:
                ax.plot(event.xdata, event.ydata, "ro")
                resetPt(ax, allDist(event.ydata, event.xdata, IO),
                        (event.ydata, event.xdata))

        # If the event is mouse right click, clean template points
        elif event.button == 3:
            cleanPt(ax)

        ax.figure.canvas.draw_idle()    # Update the canvas


def keyPress(event):
    """Handle the key press event."""
    global ptList, curPt
    if event.inaxes:    # Check if the mouse is in the image area
        ax = event.inaxes
        if event.key == 'n' and 0 not in curPt:     # Save current point
            savePt(ax.figure.get_axes())
            ax.figure.canvas.draw_idle()
            print "Point%2d has been saved" % len(ptList)

        if event.key == 'd':            # Show current point information
            print "[row, col]: [%8.3f, %8.3f]," % tuple(curPt[:2]),
            print "[x, y]: [%8.3f, %8.3f]," % tuple(curPt[2:4]),
            print "[X, Y, Z]: [%12.8f, %12.8f, %12.8f]" % tuple(curPt[4:])


def savePt(axes):
    """Store current object point and image point coordinates to point list."""
    global ptList, curPt
    ptList.append(curPt)
    curPt = [0, 0, 0, 0, 0, 0, 0]
    for ax in axes:
        ax.lines[-1].set_color('b')  # Change the color of stored point


def resetPt(ax, data, imgPt=None):
    """Reset template point."""
    # Remove old template point object
    if len(ax.lines) > 1:
        if ax.lines[-2].get_c() == 'r':
            ax.lines[-2].remove()

    # Update the current point information and the point list
    global ptList, curPt
    if ax.colNum == 0:      # For intensity image
        curPt[4:] = data
    elif ax.colNum == 1:    # For colored photo
        curPt[:2] = imgPt
        curPt[2:4] = data


def cleanPt(ax):
    """Clean template point."""
    # Clean all the template points in the subplot
    if len(ax.lines):
        if ax.lines[-1].get_c() == 'r':
            ax.lines[-1].remove()
            if ax.colNum == 0:      # For intensity image
                curPt[4:7] = (0, 0, 0)
            elif ax.colNum == 1:    # For colored photo
                curPt[:4] = (0, 0, 0, 0)


def getIO(IOFileName):
    """Read interior orientation information from file."""
    data = np.genfromtxt(IOFileName)

    # Define interior orientation parameters
    IO = {}
    keys = ["f", "xp", "yp", "Fw", "Fh", "px", "k1", "k2", "k3", "p1", "p2"]
    for key in keys:
        IO[key] = data[keys.index(key)]

    return IO


def main():
    # Read images
    intensityImg = imread('../images/Intensity100.png', 0)
    rgbImg = imread('../images/P1_L.jpg')
    outputFileName = 'result.txt'
    LiDARFileName = '../ptCloud/XYZ_edited_notree.txt'
    IOFileName = '../param/IO.txt'
    sf = 100     # Scale factor for LiDAR image

    measureProc(
        intensityImg, rgbImg, LiDARFileName, outputFileName, sf, IOFileName)

    return 0


if __name__ == '__main__':
    # Define global variables
    ptList = []
    curPt = [0, 0, 0, 0, 0, 0, 0]   # row col x y X Y Z
    main()
