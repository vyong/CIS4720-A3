import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

targetDir = sys.argv[1]

imgR = cv2.imread(os.path.join(targetDir, 'r.tif'), 0)
imgG = cv2.imread(os.path.join(targetDir, 'b.tif'), 0)
imgB = cv2.imread(os.path.join(targetDir, 'g.tif'), 0)

surf = cv2.xfeatures2d.SURF_create()

kpR, desR = surf.detectAndCompute(imgR, None)
kpG, desG = surf.detectAndCompute(imgG, None)
kpB, desB = surf.detectAndCompute(imgB, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

#clusters = np.array([desR])
#bf.add(clusters)
#bf.train()

#matches = bf.match(desG)


matches = bf.match(desR, desG)

matches = sorted(matches, key = lambda x:x.distance)
print len(matches)

imgOut = np.array([])
imgOut = cv2.drawMatches(imgR, kpR, imgG, kpG, matches[:100], imgOut, flags=2)

cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
cv2.imshow("Display", imgOut)
cv2.waitKey(0);
#print cv2.__version__
