import os
import shutil
import random

input_dir = 'C:/Shreyas/AI courseowkr/preprocessed'
output_dir = 'C:/Shreyas/AI courseowkr/my_dataset'

os.makedirs(os.path.join(output_dir, "train/mri"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "train/ct"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val/mri"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val/ct"), exist_ok=True)


patients = set(f.split("_")[0] for f in os.listdir(input_dir) if f.endswith(".nii.gz"))

# Shuffle and split into 80% train and 20% val
random.seed(42)  
patients = list(patients)
random.shuffle(patients)
split_idx = int(len(patients) * 0.8)
train_patients = patients[:split_idx]
val_patients = patients[split_idx:]

def process_files(patient_list, folder_name):
    for patient_id in patient_list:
        mri_file = f"{patient_id}_mr_resampled.nii.gz"
        if os.path.exists(os.path.join(input_dir, mri_file)):
            shutil.copy(
                os.path.join(input_dir, mri_file),
                os.path.join(output_dir, folder_name, "mri", f"mri_{patient_id}.nii.gz")
            )
        ct_file = f"{patient_id}_ct_resampled.nii.gz"
        if os.path.exists(os.path.join(input_dir, ct_file)):
            shutil.copy(
                os.path.join(input_dir, ct_file),
                os.path.join(output_dir, folder_name, "ct", f"ct_{patient_id}.nii.gz")
            )

process_files(train_patients, "train")
process_files(val_patients, "val")

print("Splitting and renaming completed!")