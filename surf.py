import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

targetDir = sys.argv[1]

imgR = cv2.imread(os.path.join(targetDir, 'r.tif'), 0)
imgG = cv2.imread(os.path.join(targetDir, 'g.tif'), 0)
imgB = cv2.imread(os.path.join(targetDir, 'b.tif'), 0)
height, width = imgR.shape[:2] # Guaranteed to be same size for easy compositing
shape = (width, height)

surf = cv2.xfeatures2d.SURF_create()

kpR, desR = surf.detectAndCompute(imgR, None)
kpG, desG = surf.detectAndCompute(imgG, None)
kpB, desB = surf.detectAndCompute(imgB, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

matches = bf.match(desR, desG)
goodMatches = [match for match in matches if match.distance < 300]

matchR = np.array([kpR[match.queryIdx].pt for match in goodMatches])
matchG = np.array([kpG[match.trainIdx].pt for match in goodMatches])
rgH, mask = cv2.findHomography(matchG, matchR, cv2.RANSAC)

matches = bf.match(desR, desB)
goodMatches = [match for match in matches if match.distance < 300]
matchR = np.array([kpR[match.queryIdx].pt for match in goodMatches])
matchB = np.array([kpB[match.trainIdx].pt for match in goodMatches])
rbH, mask = cv2.findHomography(matchB, matchR, cv2.RANSAC)

warpG = cv2.warpPerspective(imgG, rgH, shape)
warpB = cv2.warpPerspective(imgB, rbH, shape)

stacked = np.dstack((imgR, warpG, warpB))

cv2.namedWindow("warped", cv2.WINDOW_NORMAL)
cv2.imshow("warped", stacked)
cv2.waitKey(0)
