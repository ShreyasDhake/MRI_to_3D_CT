import os
import shutil
import random

base_dir = "C:/Shreyas/AI courseowkr/png_dataset"
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")
test_mri_dir = os.path.join(test_dir, "mri")
test_ct_dir = os.path.join(test_dir, "ct")

os.makedirs(test_mri_dir, exist_ok=True)
os.makedirs(test_ct_dir, exist_ok=True)

val_mri_dir = os.path.join(val_dir, "mri")
val_ct_dir = os.path.join(val_dir, "ct")

mri_images = os.listdir(val_mri_dir)

random.shuffle(mri_images)
num_test_images = int(len(mri_images) * 0.2)  
test_images = mri_images[:num_test_images]

for img in test_images:
    src_mri = os.path.join(val_mri_dir, img)
    dst_mri = os.path.join(test_mri_dir, img)
    shutil.move(src_mri, dst_mri)

    src_ct = os.path.join(val_ct_dir, img.replace("mri", "ct"))
    if os.path.exists(src_ct):
        dst_ct = os.path.join(test_ct_dir, img.replace("mri", "ct"))
        shutil.move(src_ct, dst_ct)

print(f"Moved {num_test_images} images to the test folder.")