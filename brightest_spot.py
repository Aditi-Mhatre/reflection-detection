#this one is much better (compared to detect_bright_spots.py) however this only takes note of the brightest point within the image
#need to manually add the radius and does not autodetect 

import numpy as np
import argparse
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-r", "--radius", type = int, help = "radius of Gaussian blur; must be odd")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
image = orig.copy()
x, y = maxLoc
print("Location of the brightest point ", x, ",", y)
cv2.circle(image, maxLoc, args["radius"], (255,0,0), 2)

cv2.imshow("Robust", image)
cv2.waitKey(0)
