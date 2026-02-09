import os
from PIL import Image

input_folder = "/home/chuang/hcg/projects/control_suite/temp/lab_downstairs_test_2/rgb"
output_folder = "/home/chuang/hcg/projects/control_suite/temp/lab_downstairs_test_2/rgb_jpg"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png")):
        with Image.open(os.path.join(input_folder, filename)) as img:
            # Convert to RGB (required if source is RGBA/PNG)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            img.save(os.path.join(output_folder, filename), "JPEG", optimize=True, quality=70)
            print(f"Compressed {filename}")
