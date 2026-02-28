import cv2

# load the image
img = cv2.imread('a.jpg')

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply thresholding to segment the image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# apply morphological operations to remove noise and fill gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

# find contours in the segmented image
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw the contours on the original image
for contour in contours:
    cv2.drawContours(img, [contour], 0, (0, 255, 0), 2)

# save the segmented image
cv2.imwrite('segmented.jpg', img)
