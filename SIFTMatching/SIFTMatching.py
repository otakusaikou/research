#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys


def match(fileName1, fileName2, ratio, show=False):
    """SIFT matching with opencv.

    Reference : http://goo.gl/70Tk8G
    """
    # Read image
    leftImg = cv2.imread(fileName1)
    rightImg = cv2.imread(fileName2)

    # Convert image from bgr to rgb
    leftImgRGB = cv2.cvtColor(leftImg, cv2.COLOR_BGR2RGB)
    rightImgRGB = cv2.cvtColor(rightImg, cv2.COLOR_BGR2RGB)

    # Convert image to gray scale
    leftGray = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
    rightGray = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)

    # Create sift detector object
    sift = cv2.xfeatures2d.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(leftGray, None)
    kp2, des2 = sift.detectAndCompute(rightGray, None)

    # Create Brute-Force matching object with default parameters
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    # Get coordinates of matching points
    leftPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    rightPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Apply ransac algorithm and homography model to find the inliers
    M, mask = cv2.findHomography(leftPts, rightPts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    print "Number of matching points: %d" % leftPts.shape[0]

    if show:
        # Draw matching points with green line
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor = None,
                           matchesMask=matchesMask,
                           flags = 2)

        matchImg = cv2.drawMatches(
            leftImgRGB, kp1, rightImgRGB, kp2, good, None, **draw_params)

        plt.imshow(matchImg, "gray")
        plt.show()

    matchesMask = np.array(matchesMask) == 1
    return leftPts[matchesMask], rightPts[matchesMask]


def main():
    if len(sys.argv) != 1:
        match(sys.argv[-2], sys.argv[-1], 0.8, show=True)
    else:
        match(
            "../images/P1_L_RGB50.png",
            "../images/P1_C.jpg",
            0.75,
            show=True)

    return 0

if __name__ == '__main__':
    main()
