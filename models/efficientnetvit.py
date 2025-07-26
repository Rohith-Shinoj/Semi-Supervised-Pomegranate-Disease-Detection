import torch
import torch.nn as nn
import timm

class EfficientnetViT(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        self.conv_out = nn.Conv2d(320, 768, kernel_size=1)  
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8), num_layers=4
        )
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        feats = self.backbone(x)[-1]  
        feats = self.conv_out(feats).flatten(2).permute(2, 0, 1) 
        cls_token = self.cls_token.expand(-1, x.size(0), -1)
        feats = torch.cat([cls_token, feats], dim=0)
        x = self.transformer(feats)[0]
        return self.head(x)
