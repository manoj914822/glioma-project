import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('a.jpg')

# Check if the image loaded successfully
if image is None:
    raise ValueError("Image not found. Please check the file path.")

# Convert RGB to Gray using Luminosity method
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Noise removal methods
# Median Filter
median_filtered = cv2.medianBlur(gray_image, 5)

# Gaussian Filter
gaussian_filtered = cv2.GaussianBlur(gray_image, (5, 5), 0)

# High Pass Filter
kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
high_pass_filtered = cv2.filter2D(gray_image, -1, kernel)

# Image Sharpening methods
# Unsharp Masking
blurred = cv2.GaussianBlur(gray_image, (9, 9), 10.0)
unsharp_mask = cv2.addWeighted(gray_image, 1.5, blurred, -0.5, 0)

# Laplacian Filter
laplacian_filtered = cv2.Laplacian(gray_image, cv2.CV_64F)

# Sobel Filter
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Thresholding methods
# Gaussian Thresholding
_, gaussian_thresh = cv2.threshold(gaussian_filtered, 128, 255, cv2.THRESH_BINARY)

# Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

# Segmentation methods
# Edge Based (Canny)
edges = cv2.Canny(gray_image, 100, 200)

# Region Based (Watershed)
_, binary_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dist_transform = cv2.distanceTransform(binary_thresh, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
sure_bg = cv2.dilate(binary_thresh, np.ones((3, 3), np.uint8), iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
watershed_segmented = cv2.watershed(image, markers)
image[watershed_segmented == -1] = [0, 0, 255]  # Marking boundaries in red

# Save the processed images
cv2.imwrite('gray_image.jpg', gray_image)
cv2.imwrite('median_filtered.jpg', median_filtered)
cv2.imwrite('gaussian_filtered.jpg', gaussian_filtered)
cv2.imwrite('high_pass_filtered.jpg', high_pass_filtered)
cv2.imwrite('unsharp_mask.jpg', unsharp_mask)
cv2.imwrite('laplacian_filtered.jpg', laplacian_filtered)
cv2.imwrite('sobel_combined.jpg', sobel_combined)
cv2.imwrite('gaussian_thresh.jpg', gaussian_thresh)
cv2.imwrite('adaptive_thresh.jpg', adaptive_thresh)
cv2.imwrite('edges.jpg', edges)
cv2.imwrite('watershed_segmented.jpg', image)

# Show all images in a single window
titles = [
    'RGB to Gray (Luminosity)', 'Noise Removal', 'Image Sharpening',
    'Thresholding', 'Segmentation'
]

subtitles = [
    ['Gray Image'],
    ['Median Filter', 'Gaussian Filter', 'High Pass Filter'],
    ['Unsharp Masking', 'Laplacian', 'Sobel'],
    ['Gaussian Threshold', 'Adaptive Threshold'],
    ['Edge Based (Canny)', 'Region Based (Watershed)']
]

images = [
    gray_image,
    [median_filtered, gaussian_filtered, high_pass_filtered],
    [unsharp_mask, laplacian_filtered, sobel_combined],
    [gaussian_thresh, adaptive_thresh],
    [edges, watershed_segmented]
]

plt.figure(figsize=(15, 12))

# Main Titles
for i, title in enumerate(titles):
    plt.subplot(5, 1, i + 1)
    plt.title(title)
    plt.axis('off')

    # Subtitles and images
    for j, subtitle in enumerate(subtitles[i]):
        plt.subplot(5, len(subtitles[i]), i * len(subtitles[i]) + j + 1)
        if i == 0:  # For Gray Image
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(images[i][j], cmap='gray' if i != 4 else None)
        plt.title(subtitle)
        plt.axis('off')

plt.tight_layout()
plt.show()
