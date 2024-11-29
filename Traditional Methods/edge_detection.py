import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Configure font for plot labels
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16

# Load the image
image_path = 'coins.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Canny edge detection
edges_canny = cv2.Canny(image, threshold1=100, threshold2=200)

# Sobel edge detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # X direction
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Y direction
edges_sobel = np.hypot(sobel_x, sobel_y)  # Combine the two gradients
edges_sobel = np.uint8(edges_sobel)  # Convert to uint8

# Laplacian edge detection
edges_laplacian = cv2.Laplacian(image, cv2.CV_64F)
edges_laplacian = np.uint8(np.abs(edges_laplacian))  # Convert to uint8

# Plot the results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image', fontsize=24)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(edges_canny, cmap='gray')
plt.title('Canny Edge Detection', fontsize=24)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(edges_sobel, cmap='gray')
plt.title('Sobel Edge Detection', fontsize=24)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(edges_laplacian, cmap='gray')
plt.title('Laplacian Edge Detection', fontsize=24)
plt.axis('off')

plt.tight_layout()
plt.savefig('edge_comparison.png', bbox_inches='tight')
plt.show()
