import cv2

# load the image in grayscale
img = cv2.imread('a.jpg', 0)

# apply the Canny edge detection algorithm
edges = cv2.Canny(img, 100, 200)

# save the edge detected image
image=cv2.imwrite('edges.jpg', edges)
