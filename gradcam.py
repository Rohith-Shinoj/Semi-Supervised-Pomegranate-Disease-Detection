# gradcam.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models.efficientnetvit import EfficientnetViT

model = EfficientnetViT(num_classes=3)
model.load_state_dict(torch.load("model_final.pth"))
model.eval().cuda()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).cuda()
    rgb_image = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    return img_tensor, rgb_image

target_layer = model.backbone[-1]  

cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True)

def run_gradcam(image_path, class_idx, label=""):
    input_tensor, rgb_image = load_image(image_path)
    targets = [ClassifierOutputTarget(class_idx)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(5, 5))
    plt.imshow(visualization)
    plt.axis("off")
    plt.show()

run_gradcam("bacterial/IMG023.jpg", class_idx=1, label="Bacterial")
run_gradcam("fungal/IMG045.jpg", class_idx=2, label="Fungal")
