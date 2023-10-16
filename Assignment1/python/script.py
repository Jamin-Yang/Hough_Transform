import numpy as np
import cv2
def apply_filter(input_image, filter_kernel):
    # Get the dimensions of the input image and the filter kernel
    image_height, image_width = input_image.shape
    kernel_height, kernel_width = filter_kernel.shape

    # Calculate the padding required for zero padding
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    # Create a new image with zero padding
    padded_image = np.pad(input_image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant', constant_values=0)

    # Initialize the output image
    output_image = np.zeros_like(input_image)

    # Perform convolution with zero padding
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region from the padded image
            region = padded_image[i:i + kernel_height, j:j + kernel_width]

            # Perform element-wise multiplication and sum
            output_pixel = np.sum(region * filter_kernel)

            # Store the result in the output image
            output_image[i, j] = output_pixel

    return output_image

def apply_sobel_filter(input_image):
    # Define Sobel filters for horizontal and vertical edge detection
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply Sobel filters to the input image
    gradient_x = apply_filter(input_image, sobel_x)
    gradient_y = apply_filter(input_image, sobel_y)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Calculate gradient direction
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    height, width = gradient_magnitude.shape
    suppressed_image = np.copy(gradient_magnitude)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = gradient_direction[i, j]

            # Find neighboring pixel coordinates in the gradient direction
            if (0 <= angle < np.pi / 8) or (15 * np.pi / 8 <= angle):
                prev = gradient_magnitude[i, j - 1]
                next = gradient_magnitude[i, j + 1]
            elif (np.pi / 8 <= angle < 3 * np.pi / 8):
                prev = gradient_magnitude[i - 1, j - 1]
                next = gradient_magnitude[i + 1, j + 1]
            elif (3 * np.pi / 8 <= angle < 5 * np.pi / 8):
                prev = gradient_magnitude[i - 1, j]
                next = gradient_magnitude[i + 1, j]
            else:
                prev = gradient_magnitude[i - 1, j + 1]
                next = gradient_magnitude[i + 1, j - 1]

            # Suppress non-maximum values
            if gradient_magnitude[i, j] < prev or gradient_magnitude[i, j] < next:
                suppressed_image[i, j] = 0

    return suppressed_image

def edge_detection_sobel(input_image):
    # Apply Sobel filter to obtain gradient magnitude and direction
    gradient_magnitude, gradient_direction = apply_sobel_filter(input_image)

    # Apply non-maximum suppression
    edge_image = non_maximum_suppression(gradient_magnitude, gradient_direction)

    return edge_image

# Example usage:
# Create a sample grayscale image (e.g., 5x5)

input_image = cv2.imread('./data/img01.jpg')
# Apply the Sobel edge detection filter
edge_image = edge_detection_sobel(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY))

# Display the edge-detected image
cv2.imshow('',edge_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()