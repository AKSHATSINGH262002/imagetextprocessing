
import io
import os
from google.cloud import vision
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract




def analyzing_img(image_path):
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_path)

    # Perform OCR using Tesseract
    recognized_text = pytesseract.image_to_string(image)

    # Print the recognized text
    print("Recognized Text:")
    print(recognized_text)

    return recognized_text
    
def extract_text(image_path):
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_path)

    # Perform OCR using Tesseract
    extracted_text = pytesseract.image_to_string(image)

    # Print the extracted text
    print("Extracted Text:")
    print(extracted_text)

    return extracted_text

def segmentationimg(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    # Thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Contour detection
    cont, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    segmented = cv2.drawContours(img.copy(), cont, -1, (0, 255, 0), 3)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.show()
    
    return cont

def main():
    image_path = input("Please enter the path to your image: ")

    # Check if the file exists
    if not os.path.isfile(image_path):
        print("File not found. Please check the path and try again.")
        return

    # Step 1: Analyze image
    analyzing_img(image_path)

    # Step 2: Extract text
    extract_text(image_path)

    # Step 3: Segment image
    segmentationimg(image_path)

if __name__ == "__main__":
    main()
