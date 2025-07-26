# export_onnx.py
import torch
from models.efficientnetvit import EfficientnetViT

model = EfficientnetViT(num_classes=3)
model.load_state_dict(torch.load("model_final.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=11
)
