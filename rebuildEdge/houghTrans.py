#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module can detect straight line from image."""
import cv2
import numpy as np
from scipy.misc import imread
from skimage.color import rgb2gray
from skimage import feature
import skimage.transform as st


def getStrtLine(imgFileName, cannyParam, houghParam, outputImg=False):
    """Detect straight lines."""
    # Read image
    img = imread(imgFileName)
    gray = rgb2gray(img) * 255

    # Detect lines
    isEdge = feature.canny(gray, *cannyParam)
    lines = st.probabilistic_hough_line(edges, *houghParam)

    if outputImg:
        # For edge detection
        edgeImg = np.zeros(img.shape)
        edgeImg[isEdge] = 255
        cv2.imwrite("Edge.jpg", edgeImg)

        # For result of hough transformation
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for line in lines:
            cv2.line(img, line[0], line[1], (0, 255, 0), 2)
        cv2.imwrite("HoughLines.jpg", img)

    return lines


def main():
    # Define file name
    imgFileName = '../images/P2_R.jpg'

    # Define parameters for canny edge detector and hough transformation
    cannySigma = 2          # Standard deviation of the Gaussian filter
    cannyLowThres = 1       # Lower bound for hysteresis thresholding
    cannyHighThres = 25     # Upper bound for hysteresis thresholding
    houghThres = 10     # The minimum number of intersections to detect a line
    houghLineLength = 30    # Minimum accepted length of detected lines
    houghLineGap = 1        # Maximum gap between pixels to still form a line

    lines = getStrtLine(
        imgFileName,
        (cannySigma, cannyLowThres, cannyHighThres),
        (houghThres, houghLineLength, houghLineGap), True)

    for l in lines:
        print "START(row, col) - END(row, col): (%d, %d) - (%d, %d)" % \
            (l[0][1], l[0][0], l[1][1], l[1][0])


if __name__ == '__main__':
    main()
