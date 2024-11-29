import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure font
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16

# Load the image in grayscale
image_path = "./coins.jpg"  # Replace with the path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Otsu's thresholding
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply Mean adaptive thresholding
mean_thresh = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)

# Apply Gaussian adaptive thresholding
gaussian_thresh = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# Plot the images with labels
titles = ["Original Image", "Otsu Thresholding", "Mean Thresholding", "Gaussian Thresholding"]
images = [image, otsu_thresh, mean_thresh, gaussian_thresh]

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i], fontsize=20)
    plt.axis('off')

plt.tight_layout()
plt.savefig('thresholding_comparison.png', bbox_inches='tight')
plt.show()


