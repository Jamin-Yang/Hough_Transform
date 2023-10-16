import numpy as np
import cv2


def myImageFilter(img0, h):

    # Check if the image was loaded successfully
    if img0 is None:
        print('Image not found or could not be opened.')
    # # set channel to 1
    try:
        # turn RGB image to GRAY image
        height, width, channels = img0.shape[:3]
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    except ValueError:
        # GRAY image
        height, width = img0.shape[:2]

    # zero padding, suppose kernel is a square matrix
    kernel_height, kernel_width = h.shape
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2
    padded_image = np.pad(img0, ((padding_height, padding_height), 
                                        (padding_width, padding_width)), mode='constant', constant_values=0)

    # convolution
    img1 = np.zeros((height,width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            img1[i,j] = np.sum(region * h)

    return img1


if __name__ == '__main__':
    path = './data/img01.jpg'
    h = np.array([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]])/9
    img0 = cv2.imread(path)
    img1 = myImageFilter(img0=img0, h=h)
    cv2.namedWindow('origin')
    cv2.namedWindow('blur')
    cv2.imshow('origin', img0)
    cv2.imshow('blur', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()