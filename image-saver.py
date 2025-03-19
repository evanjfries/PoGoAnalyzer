import numpy as np
from PIL import Image
import os

# === Set file paths here ===
npy_path = "src/references/Loss.npy"  # Replace with your .npy file path
output_path = "src/references/Loss.png"  # Replace with desired output image path (e.g., "image.png")

# === Load .npy file ===
array = np.load(npy_path)

# Normalize to 0–255 if not already uint8
if array.dtype != np.uint8:
    array_min = array.min()
    array_max = array.max()
    if array_max > array_min:
        array = 255 * (array - array_min) / (array_max - array_min)
    else:
        array = np.zeros_like(array)
    array = array.astype(np.uint8)

# Determine image mode (grayscale or RGB/RGBA)
if array.ndim == 2:
    mode = 'L'  # Grayscale
elif array.ndim == 3 and array.shape[2] == 3:
    mode = 'RGB'
elif array.ndim == 3 and array.shape[2] == 4:
    mode = 'RGBA'
else:
    raise ValueError(f"Unsupported array shape for image conversion: {array.shape}")

# Convert array to image
img = Image.fromarray(array, mode)

# Save image
img.save(output_path)
print(f"Image saved to {output_path}")
