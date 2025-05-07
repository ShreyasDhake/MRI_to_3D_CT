import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image
import pyvista as pv

# Define paths
#fake_B_dir = "C:/Shreyas/AI courseowkr/pytorch-CycleGAN-and-pix2pix/results/mri_to_ct_pix2pix/test_latest/images/fake_B"
fake_B_dir = "C:/Shreyas/AI courseowkr/results1"
#fake_B_dir = "C:/Shreyas/AI courseowkr/png_dataset_test/test/mri"
# Get list of PNG files in the directory
png_files = sorted([f for f in os.listdir(fake_B_dir) if f.endswith('.png')])

# Initialize a list to hold the image slices
image_slices = []

# Load each PNG file and append it to the list
for png_file in png_files:
    img_path = os.path.join(fake_B_dir, png_file)
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    image_slices.append(img_array)

# Stack the slices to create a 3D structure
image_3d = np.stack(image_slices, axis=0)

# Calculate metrics for the 3D structure
mean_intensity = np.mean(image_3d)
std_intensity = np.std(image_3d)
min_intensity = np.min(image_3d)
max_intensity = np.max(image_3d)
volume = np.sum(image_3d > 0)  # Assuming non-zero values represent the volume

print(f"Mean Intensity: {mean_intensity}")
print(f"Standard Deviation of Intensity: {std_intensity}")
print(f"Minimum Intensity: {min_intensity}")
print(f"Maximum Intensity: {max_intensity}")
print(f"Volume (number of non-zero voxels): {volume}")

# Convert the 3D numpy array to a SimpleITK image
sitk_image = sitk.GetImageFromArray(image_3d)

# Display the 3D structure using matplotlib
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image_3d[:, :, image_3d.shape[2] // 2], cmap='gray')
ax[0].set_title('Axial View')
ax[1].imshow(image_3d[:, image_3d.shape[1] // 2, :], cmap='gray')
ax[1].set_title('Coronal View')
ax[2].imshow(image_3d[image_3d.shape[0] // 2, :, :], cmap='gray')
ax[2].set_title('Sagittal View')
#plt.show()

# Create a 3D visualization using pyvista
volume = pv.wrap(image_3d)
plotter = pv.Plotter()
plotter.add_volume(volume, cmap="gray", opacity="sigmoid")
plotter.show()

print("3D structure created and displayed.")