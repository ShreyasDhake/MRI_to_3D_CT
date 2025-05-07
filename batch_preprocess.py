import os
import subprocess

input_base = 'C:/Shreyas/AI courseowkr/Task1/Task1/pelvis'
output_base = 'C:/Shreyas/AI courseowkr/preprocessed'

os.makedirs(output_base, exist_ok=True)

for folder in os.listdir(input_base):
    patient_path = os.path.join(input_base, folder)
    if os.path.isdir(patient_path):  
        mr_input = os.path.join(patient_path, "mr.nii.gz")
        ct_input = os.path.join(patient_path, "ct.nii.gz")
        
        mr_output = os.path.join(output_base, f"{folder}_mr_resampled.nii.gz")
        ct_output = os.path.join(output_base, f"{folder}_ct_resampled.nii.gz")


        if os.path.exists(mr_input):
            try:
                subprocess.run([
                    "python", "pre_process_tools.py", "resample",
                    "--i", mr_input,
                    "--o", mr_output,
                    "--s", "1", "1", "2.5"
                ], check=True)
                print(f"Processed MRI for {folder}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing MRI for {folder}: {e}")

        if os.path.exists(ct_input):
            try:
                subprocess.run([
                    "python", "pre_process_tools.py", "resample",
                    "--i", ct_input,
                    "--o", ct_output,
                    "--s", "1", "1", "2.5"
                ], check=True)
                print(f"Processed CT for {folder}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing CT for {folder}: {e}")