
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np


weak_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

strong_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomResizedCrop(224),
    transforms.RandomErasing(p=0.5),
    transforms.ToTensor()
])

## Segmentation model

seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True).eval()

seg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_segmentation_mask(image: Image.Image, threshold: float = 0.7):
    img_tensor = seg_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = seg_model(img_tensor)['out']
    mask = output.squeeze(0).argmax(0).byte().numpy()
    return Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size)

def remove_background(image: Image.Image, threshold: float = 0.7):
    mask = get_segmentation_mask(image, threshold)
    mask_np = np.array(mask) > 0 

    img_np = np.array(image).copy()
    img_np[~mask_np] = 255
    return Image.fromarray(img_np)
