import cv2
import numpy as np

# Load the image
img = cv2.imread('currency.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define a function to calculate the aspect ratio of the contours
def aspect_ratio(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return float(w) / h

# Define a function to detect fake currency based on aspect ratio
def detect_fake_currency(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours in the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the aspect ratio of each contour and check if it is outside the normal range
    for cnt in contours:
        ar = aspect_ratio(cnt)
        if ar < 1.2 or ar > 2.5:
            return True

    return False

# Loop over the contours and draw them on the original image
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if detect_fake_currency(img[y:y+h, x:x+w]):
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image
cv2.imshow('Fake Currency Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
