import sys
import os
import cv2
import numpy as np

targetDir = sys.argv[1]

imgR = cv2.imread(os.path.join(targetDir, 'r.tif'), 0)
imgG = cv2.imread(os.path.join(targetDir, 'g.tif'), 0)
imgB = cv2.imread(os.path.join(targetDir, 'b.tif'), 0)
bad = np.dstack((imgR, imgG, imgB))

cv2.namedWindow("warped", cv2.WINDOW_NORMAL)
cv2.imshow("warped", bad)
cv2.waitKey(0)
