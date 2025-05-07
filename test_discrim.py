import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
from discrimin_model import Discriminative_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_mri_dir = "C:/Shreyas/AI courseowkr/png_dataset_test/test/mri"
output_ct_dir = "C:/Shreyas/AI courseowkr/results1"

os.makedirs(output_ct_dir, exist_ok=True)

model = Discriminative_model().to(device)
model.load_state_dict(torch.load("Discriminative_model_final.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_images = sorted(os.listdir(test_mri_dir))

for img_name in test_images:
    img_path = os.path.join(test_mri_dir, img_name)
    
    image = Image.open(img_path)  
    input_tensor = transform(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = output_tensor.squeeze().cpu().numpy()  
    output_image = (output_image * 127.5 + 127.5).astype(np.uint8)  

    output_img_path = os.path.join(output_ct_dir, f"ct_{img_name}")
    Image.fromarray(output_image).save(output_img_path)

    print(f"Processed: {img_name} -> {output_img_path}")

print("Testing complete! CT images saved in:", output_ct_dir)