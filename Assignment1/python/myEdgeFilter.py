import numpy as np
from scipy import signal    # For signal.gaussian function
import cv2
from myImageFilter import myImageFilter

def myNMS(gradient_magnitude, gradient_direction):
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



def myEdgeFilter(img0, sigma):
    # calculate kernel
    k_size = 2 * np.ceil(3*sigma) + 1
    kernel = np.outer(signal.gaussian(k_size,sigma),signal.gaussian(k_size,sigma)) # signal.gaussian only generate 1D array

    # gaussian filter
    img1 = myImageFilter(img0=img0, h=kernel)

    # define sobel filter
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    gradient_x_my = myImageFilter(img1, sobel_x)
    gradient_y_my = myImageFilter(img1, sobel_y)  # !!!!it was uint8, will cause overflow

    
    temp_x = np.array(gradient_x_my, dtype=np.int16)
    temp_y = np.array(gradient_y_my, dtype=np.int16)

    direction = np.arctan2(gradient_y_my, gradient_x_my)
    # gradient = np.sqrt(gradient_x_my**2 + gradient_y_my**2)
    gradient = np.sqrt(temp_x**2 + temp_y**2)
    gradient = myNMS(gradient, direction)
    # # Apply a threshold to retain strong edges
    threshold = 100
    slim_edges = np.where(gradient > threshold, 255, 0).astype(np.uint8)

    cv2.namedWindow('z')
    cv2.imshow('z', slim_edges)
    # cv2.imshow('canny', gradient)
    cv2.imshow('canny', cv2.Canny(image=img1,threshold1=100,threshold2=150))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = './data/img02.jpg'

    img0 = cv2.imread(path)
    
    myEdgeFilter(img0=img0, sigma=0.3)
 