from skimage import feature, exposure
import cv2
image = cv2.imread('sobel.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fd, hog_image = feature.hog(image,orientations=3,pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys',feature_vector=True,visualize=True)
cv2.imshow('hog', hog_image)
cv2.waitKey(0)