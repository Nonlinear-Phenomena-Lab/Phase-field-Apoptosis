import cv2
import numpy as np

def binarize_image(image_path, output_path, threshold=200):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_path, binary_image)

image_paths = [
    'image_1.png',
    'image_2.png',
    'image_3.png',
    'image_4.png',
    'image_7.png',
    'image_8.png'
]

output_paths = [
    'output_image1.png',
    'output_image2.png',
    'output_image3.png',
    'output_image4.png',
    'output_image7.png',
    'output_image8.png'
]

for input_path, output_path in zip(image_paths, output_paths):
    binarize_image(input_path, output_path)
