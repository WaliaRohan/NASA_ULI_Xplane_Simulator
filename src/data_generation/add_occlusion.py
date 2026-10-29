import os
import numpy as np
from PIL import Image, ImageDraw

# Configuration
DATA_DIR = os.environ['NASA_DATA_DIR'] 
dataset_type = 'large_images' # large_images or downsampled
condition = 'afternoon' # 'afternoon', 'morning', 'night', 'overcast'

input_dir = os.path.join(DATA_DIR, dataset_type, condition, f"{condition}_test") # Source image files
output_dir = os.path.join(DATA_DIR, dataset_type, f"{condition}_occ", f"{condition}_occ_test") # Save dir for occluded images

percentage_occ = 10 # How many images to add occlusion to
occ_rect_size_perc = 0.5 # How large occlusion rectangle be wrt image size

os.makedirs(output_dir, exist_ok=True)

# Get image files in sorted order
image_files = sorted([
    f for f in os.listdir(input_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
])

if not image_files:
    raise ValueError((f"No images found in {input_dir}"))

# Select uniformmly spaced images, including first and last
n_selected = max(1, int(round(len(image_files)*percentage_occ/100)))
indices = np.linspace(0, len(image_files) - 1, n_selected, dtype=int)

# Apply occlusion and save
for idx in indices:
    filename = image_files[idx]
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    with Image.open(input_path) as img:
        img = img.convert("RGB")
        img_width, img_height = img.size

        # Occlusion rectangle
        rect_width = img_width * occ_rect_size_perc
        rect_height = img_height * occ_rect_size_perc

        rect_left = (img_width - rect_width) // 2
        rect_top = (img_height - rect_height) // 2
        rect_right = rect_left + rect_width
        rect_bottom = rect_top + rect_height

        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (rect_left, rect_top, rect_right - 1, rect_bottom -1),
            fill = (0, 0, 0)
        )

        img.save(output_path)

print(f"Total images: {len(image_files)}")
print(f"Occluded images: {len(indices)}")
print(f"Output directory: {output_dir}")