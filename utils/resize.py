import os
from PIL import Image

# Define the target dimensions
WIDTH = 106
HEIGHT = 80

# Create the 'resized' directory if it doesn't exist
if not os.path.exists('resized'):
    os.makedirs('resized')

# Function to crop the center of the image
def crop_center(img, crop_width, crop_height):
    img_width, img_height = img.size
    return img.crop((
        (img_width - crop_width) // 2,
        (img_height - crop_height) // 2,
        (img_width + crop_width) // 2,
        (img_height + crop_height) // 2
    ))

# Loop through all files in the current directory
for filename in os.listdir('.'):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        try:
            # Open an image file
            with Image.open(filename) as img:
                # Resize the image while maintaining aspect ratio
                img.thumbnail((WIDTH, HEIGHT), Image.LANCZOS)
                # Crop the center of the image to the target size
                cropped_img = crop_center(img, WIDTH, HEIGHT)
                # Save the cropped image to the 'resized' folder
                cropped_img.save(os.path.join('resized', filename))
        except Exception as e:
            print(f"Could not crop and save image '{filename}'. Make sure that the file extension matches the image format.")
            print(f"Error message: {e}")
            exit(-1)

print("All images have been resized, cropped to the center, and saved in the 'resized' folder.")