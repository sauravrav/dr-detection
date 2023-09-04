import os
import cv2
from PIL import Image
print(os.getcwd())

def resize_images(folder_path, output_folder_path, target_resolution):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        resized_image = image.resize(target_resolution)
        resized_image.save(os.path.join(output_folder_path, image_file))

def remove_black_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        grayscale_image = image.convert("L")
        is_black = all(pixel == 0 for pixel in grayscale_image.getdata())

        if is_black:
            os.remove(image_path)

def rotate_and_mirror_images(folder_path, output_folder_path):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        # Rotate and mirror the image
        rotated_image_90 = image.rotate(90)
        rotated_image_120 = image.rotate(120)
        rotated_image_180 = image.rotate(180)
        rotated_image_270 = image.rotate(270)
        mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Save the rotated and mirrored images
        rotated_image_90.save(os.path.join(output_folder_path, image_file + "_rotated_90.jpg"))
        rotated_image_120.save(os.path.join(output_folder_path, image_file + "_rotated_120.jpg"))
        rotated_image_180.save(os.path.join(output_folder_path, image_file + "_rotated_180.jpg"))
        rotated_image_270.save(os.path.join(output_folder_path, image_file + "_rotated_270.jpg"))
        mirrored_image.save(os.path.join(output_folder_path, image_file + "_mirrored.jpg"))

def apply_denoising_clahe(folder_path):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply CLAHE for denoising
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        denoised_image = clahe.apply(image)

        # Save the denoised image
        cv2.imwrite(image_path, denoised_image)

# Set the paths and parameters
input_folder_path = "validate/NO_DR"
output_folder_path = "T3/validate/NO_DR"
target_resolution = (224, 224)  # or (256, 256)

# Resize the images
resize_images(input_folder_path, output_folder_path, target_resolution)

# Remove black images
remove_black_images(output_folder_path)

# Rotate and mirror images
rotate_and_mirror_images(output_folder_path, output_folder_path)

# Apply denoising with CLAHE
apply_denoising_clahe(output_folder_path)
