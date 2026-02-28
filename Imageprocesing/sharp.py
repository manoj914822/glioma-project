import cv2
import numpy as np

# load the image
img = cv2.imread('a.jpg')

# create the sharpening kernel
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])

# apply the sharpening kernel to the image
sharpened = cv2.filter2D(img, -1, kernel_sharpening)

# save the sharpened image
cv2.imwrite('sharpened.jpg', sharpened)
