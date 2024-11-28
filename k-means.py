import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('fruits.PNG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
vector = image.reshape((-1,3))
vector = np.float32(vector)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 6
attempts = 10
ret, label, center = cv2.kmeans(vector, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_img = res.reshape((image.shape))

figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1), plt.imshow(image)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2), plt.imshow(result_img)
plt.title('Segmented with k= %i' % k), plt.xticks([]), plt.yticks([])
plt.show()
