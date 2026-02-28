import cv2

# load the image
img = cv2.imread('a.jpg')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding to segment the image
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# save the thresholded image
image=cv2.imwrite('thresholded.jpg', thresh)
cv2.imshow("demo",image)
