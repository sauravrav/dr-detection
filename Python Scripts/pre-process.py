import cv2
import os
import numpy as np
import shutil

# Load images from the directory
directory = 'DR'
images = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(directory, filename))
        if img is not None:
            images.append(img)

# Process each image
# Process each image
for img, filename in zip(images, os.listdir(directory)):
    # Convert the image to grayscale and produce the mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY)

    # Find the bounding rectangle of the mask and crop the image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    crop = img[y:y+h, x:x+w]

    # Resize the cropped image and create a circular mask
    size = max(w, h)
    resize = cv2.resize(crop, (size, size))

    mask = np.zeros((size, size), dtype="uint8")
    cv2.circle(mask, (size//2, size//2), size//2, 255, -1)

    # Combine the resized image with the circular mask and remove the black border again
    circular = cv2.bitwise_and(resize, resize, mask=mask)

    gray = cv2.cvtColor(circular, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    final = circular[y:y+h, x:x+w]

    # Convert the image to HSV format
    hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)

    # Split the HSV image into Hue, Saturation and Value channels
    h, s, v = cv2.split(hsv)

    # Create a CLAHE object (Arguments are optional)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Equalize the histogram of the Hue and Value channels using CLAHE
    h_eq = clahe.apply(h)
    v_eq = clahe.apply(v)

    # Merge the equalized Hue, original Saturation and equalized Value channels
    hsv_eq = cv2.merge([h_eq, s, v_eq])

    # Convert the equalized HSV image back to RGB format
    final_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    # Save the processed image to the new directory
    new_directory = 'DR-train'
    os.makedirs(new_directory, exist_ok=True)
    cv2.imwrite(os.path.join(new_directory, filename), final_eq)
    # shutil.move(os.path.join(directory, filename), os.path.join(new_directory, filename))
print(len(images))
