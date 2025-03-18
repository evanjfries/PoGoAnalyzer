import numpy as np
import os

# Directory containing .npy files
npy_dir = "src/Opponent Team-npy"
npy_files = [f for f in os.listdir(npy_dir) if f.endswith(".npy")]

if not npy_files:
    raise FileNotFoundError(f"No .npy files found in '{npy_dir}'")

# Load and sum all .npy files
sum_array = None
file_count = 0

for file_name in npy_files:
    file_path = os.path.join(npy_dir, file_name)
    npy_data = np.load(file_path)

    if sum_array is None:
        sum_array = npy_data.astype(np.float32)
    else:
        if npy_data.shape != sum_array.shape:
            raise ValueError(f"Shape mismatch: {npy_data.shape} vs {sum_array.shape} in file {file_name}")
        sum_array += npy_data.astype(np.float32)

    file_count += 1

# Compute the average
average_array = (sum_array / file_count).astype(np.uint8)

# Save the result
output_path = os.path.join("src/references", "OpponentTeam.npy")
np.save(output_path, average_array)

print(f"✅ Averaged array from {file_count} files saved to {output_path}")
