import os
import nibabel as nib
import numpy as np
from PIL import Image

input_dir = 'C:/Shreyas/AI courseowkr/my_dataset'
output_dir = 'C:/Shreyas/AI courseowkr/pytorch-CycleGAN-and-pix2pix/datasets/png_dataset'

os.makedirs(os.path.join(output_dir, "train/mri"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train/ct"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val/mri"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val/ct"), exist_ok=True)

def nii_to_png(input_path, output_path):
    nii = nib.load(input_path)
    data = nii.get_fdata()

    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255.0
    data = data.astype(np.uint8)

    # Use the basename of the input NIfTI file for unique naming.
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    for i in range(data.shape[2]):
        img = Image.fromarray(data[:, :, i])
        save_path = os.path.join(output_path, f"{base_name}_slice{i:03d}.png")
        img.save(save_path)

for split in ["train", "val"]:
    for modality in ["mri", "ct"]:
        in_path = os.path.join(input_dir, split, modality)
        out_path = os.path.join(output_dir, split, modality)
        for file in os.listdir(in_path):
            if file.endswith(".nii.gz"):
                nii_to_png(os.path.join(in_path, file), out_path)

print("Conversion to PNG completed!")