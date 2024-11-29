#simple thresholding works better in this case as it only highlights the reflection regions = not suitable for smaller images
#otsu thresholding does not work well - discarded
#mean thresholding overall good for segmentation but not for detecting only reflection
#gaussian thresholding same as mean thresholding

import argparse
import cv2
import matplotlib.pyplot as plt
import keras
from keras import layers
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
args = vars(ap.parse_args())
path = args["image"]
img_name = path[11:15]
#print(img_name)

# load the image and display it
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
# convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# apply simple thresholding with a hardcoded threshold value
(T, threshInv) = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)
cv2.imshow("Simple Thresholding", threshInv)
cv2.imwrite('inspection/' + img_name + '_h.png',threshInv)
cv2.waitKey(0)


# instead of manually specifying the threshold value, we can use
# adaptive thresholding to examine neighborhoods of pixels and
# adaptively threshold each neighborhood
thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
cv2.imshow("Mean Adaptive Thresholding", thresh)
cv2.waitKey(0)

# perform adaptive thresholding again, this time using a Gaussian
# weighting versus a simple mean to compute our local threshold
# value
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
cv2.imshow("Gaussian Adaptive Thresholding", thresh)
cv2.waitKey(0)

