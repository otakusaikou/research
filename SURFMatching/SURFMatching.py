#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys


def match(leftImg, rightImg, ratio, show=False):
    """SURF matching with opencv.

    Reference : http://goo.gl/dhnm8n
    """
    # Convert image to gray scale
    leftGray = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
    rightGray = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)

    # Create surf detector object
    surf = cv2.xfeatures2d.SURF_create(400, 5, 5)

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(leftGray, None)
    kp2, des2 = surf.detectAndCompute(rightGray, None)

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
            leftImg, kp1, rightImg, kp2, good, None, **draw_params)

        plt.imshow(matchImg, "gray")
        plt.show()


def main():
    if len(sys.argv) != 1:
        fileName1 = sys.argv[-2]
        fileName2 = sys.argv[-1]
    else:
        fileName1 = "../images/P1_L.jpg"
        fileName2 = "../images/P1_C.jpg"

    # Read image
    leftImg = cv2.imread(fileName1)
    rightImg = cv2.imread(fileName2)

    # Convert image from bgr to rgb
    leftImgRGB = cv2.cvtColor(leftImg, cv2.COLOR_BGR2RGB)
    rightImgRGB = cv2.cvtColor(rightImg, cv2.COLOR_BGR2RGB)

    match(leftImgRGB, rightImgRGB, 0.8, show=True)

    return 0

if __name__ == '__main__':
    main()
