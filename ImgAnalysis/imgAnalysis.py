#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np


rgbImg = None
hsvImg = None
intensityMap = None
mode = None


def getLine(start, end):
    """Bresenhams' Line Algorithm.

    Return the a seris of image coordinates representing the line between
    two given points.
    """
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return np.hsplit(np.array(points), 2)


def mousePress(event):
    """Handle the mouse press event."""
    if event.inaxes:    # Check if the mouse is in the image area
        ax = event.inaxes

        # If the event is mouse right click, clean the canvas
        if event.button == 3:
            clean(ax.figure.get_axes())

        # Draw the first point when user hold the control key
        elif event.button == 1 and event.key == "control":
            clean(ax.figure.get_axes())
            if ax.numRows == 1 and ax.numCols == 2:
                ax.plot(event.xdata, event.ydata, "ro")
        ax.figure.canvas.draw_idle()    # Update the canvas


def mouseRelease(event):
    """Handle the mouse release event."""
    if event.inaxes:    # Check if the mouse is in the image area
        ax = event.inaxes

        # Draw the sample line and the second point when user hold
        # the control key
        if event.button == 1 and event.key == "control":
            if len(ax.lines) == 4:  # Clean old points and lines
                map(lambda e: e.remove(), ax.lines[:3])
            ax.plot(event.xdata, event.ydata, "ro")
            ax.plot([ax.lines[0]._x[0], event.xdata],
                    [ax.lines[0]._y[0], event.ydata], "r-")

            # Calculate the sample point
            start = tuple(map(int, ax.lines[0]._xy[0]))
            end = tuple(map(int, ax.lines[1]._xy[0]))
            xi, yi = getLine(start, end)

            # Mark the pixel through the line pass
            ax.plot(xi, yi, "rx")

            # Update the function plot
            update(ax.figure.get_axes()[1:], xi, yi, mode)

            ax.figure.canvas.draw_idle()    # Update the canvas


def clean(axes):
    """Clean all the objects in given axes."""
    for ax in axes:
        while ax.lines:
            ax.lines[0].remove()
            if ax.legend():
                ax.legend().remove()


def update(axes, xi, yi, mode):
    """Update the function plot."""
    points = np.dstack((yi, xi)).reshape(-1, 2)
    # Get red, green, blue and intensity values
    I = np.array(
        map(lambda (row, col): intensityMap[row, col], points)).reshape(-1, 1)

    R, G, B = np.hsplit(np.array(
        map(lambda (row, col): rgbImg[row, col], points)), 3)

    # Convert the value from red-green-blue to hue-saturation-value color space
    H, S, V = cv2.split(cv2.cvtColor(cv2.merge((B, G, R)), cv2.COLOR_BGR2HSV))
    clean(axes)

    # For function plot
    colors = np.array(["m", "r", "g", "b", "salmon", "lime", "cyan"])
    values = np.array([I, R, G, B, H, S, V])
    labels = np.array([
        "Intensity", "Red", "Green", "Blue", "Hue", "Saturation", "Value"])

    mode = np.array(list(mode)) == "1"

    for c, v, l in zip(colors[mode], values[mode], labels[mode]):
        axes[0].plot(yi, v, c, label=l)
        axes[1].plot(xi, v, c, label=l)

    # For legend
    axes[0].legend(handles=axes[0].lines, loc=4, fontsize="small")
    axes[1].legend(handles=axes[1].lines, loc=4, fontsize="small")


def showResult(rgbImg, intensityMap, mode):
    """Display the difference between rgb image and intensity map."""
    # Create figure
    fig = plt.figure("Result", figsize=(17, 6))
    fig.canvas.draw_idle()

    # Show the rgb image
    ax1 = fig.add_subplot(121)
    plt.title("RGB image", size=15)
    ax1.axis([0, rgbImg.shape[1], rgbImg.shape[0], 0])
    ax1.axis("off")
    ax1.imshow(rgbImg, interpolation="none")

    # Plot r, g, b and intensity values as function of row
    ax2 = fig.add_subplot(222)
    plt.title("Intensity variation plot: function of row", size=15)
    ax2.axis([0, rgbImg.shape[0], 0, 255])
    ax2.grid()

    # Plot r, g, b and intensity values as function of row
    ax3 = fig.add_subplot(224)
    plt.title("Intensity variation plot: function of column", size=15)
    ax3.axis([0, rgbImg.shape[1], 0, 255])
    ax3.grid()

    fig.tight_layout(pad=3, w_pad=5)

    # Bind the mouse event
    fig.canvas.mpl_connect("button_press_event", mousePress)
    fig.canvas.mpl_connect("button_release_event", mouseRelease)

    plt.show()


def main():
    rgbImgName = "../images/RGB.png"
    intensityMapName = "../images/Intensity.png"

    # Read image and convert it from BGR to RGB
    global rgbImg, hsvImg, intensityMap, mode
    rgbImg = cv2.cvtColor(cv2.imread(rgbImgName), cv2.COLOR_BGR2RGB)
    hsvImg = cv2.cvtColor(cv2.imread(rgbImgName), cv2.COLOR_BGR2HSV)
    intensityMap = cv2.imread(intensityMapName, 0)
    mode = "1000111"

    showResult(rgbImg, intensityMap, mode)

    return 0

if __name__ == '__main__':
    main()
