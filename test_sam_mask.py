import requests
import base64
from PIL import Image
import numpy as np
import io
import time

SAM_API_URL = "http://localhost:5000/sam/predict_with_bbox"
IMAGE_URL = "https://t3.ftcdn.net/jpg/05/82/56/80/360_F_582568095_j49qzM3AIbjr0GlNPOHRUJkfUuqVukuI.jpg"

# Bounding boxes for 4 cats
cat_bboxes = [
    [40, 100, 130, 300],
    [130, 100, 220, 300],
    [220, 100, 310, 300],
    [310, 100, 400, 300],
]

# Bounding box for surface/log (adjust if needed)
surface_bbox = [0, 300, 640, 360]

# Bounding box for background (top part, excluding cats)
background_bbox = [0, 0, 640, 100]

# Color mapping for regions
REGION_COLORS = {
    "cat": (0, 255, 0),         # green
    "surface": (0, 0, 255),     # blue
    "background": (255, 0, 0),  # red
}

# Load original image
original_image = Image.open(io.BytesIO(requests.get(IMAGE_URL).content)).convert("RGBA")
overlay = Image.new("RGBA", original_image.size)

# Decode a binary mask from SAM API response
def decode_mask(mask_b64):
    try:
        mask_bytes = base64.b64decode(mask_b64)
        return Image.open(io.BytesIO(mask_bytes)).convert("L")
    except Exception as e:
        print(f"Error decoding mask: {e}")
        return None

# Helper to send request and apply overlay
def apply_mask(bbox, label):
    payload = {
        "image": IMAGE_URL,
        "bbox": bbox
    }
    try:
        response = requests.post(SAM_API_URL, json=payload, timeout=(10, 120))
        response.raise_for_status()
        data = response.json()
        if "masks" not in data or len(data["masks"]) == 0:
            print(f"No mask returned for {label}")
            return
        mask_info = data["masks"][0]
        mask_img = decode_mask(mask_info["mask"])
        if mask_img is None:
            return
        # Resize to full image size
        mask_img = mask_img.resize(original_image.size)
        # Colorize and alpha blend
        color = REGION_COLORS[label]
        color_layer = Image.new("RGBA", original_image.size, color + (0,))
        alpha = mask_img.point(lambda p: 150 if p > 0 else 0)
        color_layer.putalpha(alpha)
        global overlay
        overlay = Image.alpha_composite(overlay, color_layer)
        print(f"Applied mask for {label}")
    except Exception as e:
        print(f"Failed to apply mask for {label}: {e}")

# Apply cat masks
for bbox in cat_bboxes:
    apply_mask(bbox, "cat")

# Apply surface mask
apply_mask(surface_bbox, "surface")

# Apply background mask
apply_mask(background_bbox, "background")

# Composite overlay and save
final = Image.alpha_composite(original_image, overlay)
filename = f"segmented_3regions_{int(time.time())}.png"
final.save(filename)
print(f"Saved segmented image as {filename}")
