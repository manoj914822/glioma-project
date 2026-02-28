import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

def plot_matrix(matrix, title):
    fig, ax = plt.subplots()
    ax.axis('off')
    tb = Table(ax, bbox=[0, 0, 1, 1])
    n_rows, n_cols = matrix.shape[:2]  # Only take the first two dimensions

    width, height = 1.0 / n_cols, 1.0 / n_rows
    for i in range(n_rows):
        for j in range(n_cols):
            # Format BGR values as a string
            bgr_values = matrix[i, j]  # This will be a 3-element array (B, G, R)
            tb.add_cell(i, j, width, height, text=str(bgr_values),
                        loc='center', facecolor='lightgray')

    ax.add_table(tb)
    plt.title(title)
    plt.show()

# Load the image
image = cv2.imread('a.jpg')  # Replace with your actual image path

# Resize the image to 5x5 pixels to create a 5x5 matrix
resized_image = cv2.resize(image, (5, 5))

# Extract the BGR values as a matrix of shape (5, 5, 3)
bgr_matrix = resized_image  # Shape (5, 5, 3)
print("BGR Matrix (5x5):")
plot_matrix(bgr_matrix, "BGR Matrix (5x5)")

# Convert BGR to Grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
gray_matrix = gray_image.astype(int)

# Display the grayscale matrix in integer format
print("\nGrayscale Matrix (5x5):")
plot_matrix(gray_matrix, "Grayscale Matrix (5x5)")
