# train_ssl.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.efficientnetvit import EfficientnetViT
from utils import weak_aug, strong_aug, remove_bg

def fixmatch_loss(model, x_l, y_l, x_u_w, x_u_s, threshold=0.95, lambda_u=1.0):
    logits_l = model(x_l)
    Lx = F.cross_entropy(logits_l, y_l)

    with torch.no_grad():
        pseudo_labels = torch.softmax(model(x_u_w), dim=-1)
        max_probs, targets_u = torch.max(pseudo_labels, dim=-1)
        mask = max_probs.ge(threshold).float()

    logits_u_s = model(x_u_s)
    Lu = F.cross_entropy(logits_u_s, targets_u, reduction='none')
    Lu = (Lu * mask).mean()

    return Lx + lambda_u * Lu

model = EfficientnetViT(num_classes=3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

labeled_loader = DataLoader(...)
unlabeled_loader = DataLoader(...)

for epoch in range(50):
    model.train()
    for (x_l, y_l), (x_u, _) in zip(labeled_loader, unlabeled_loader):
        x_l_bg_removed = [remove_bg(img) for img in x_l] 
        x_l_aug = torch.stack([weak_aug(img) for img in x_l_bg_removed]).cuda()
        y_l = y_l.cuda()
        x_u_bg_removed = [remove_bg(img) for img in x_u]
        x_u_w = torch.stack([weak_aug(img) for img in x_u_bg_removed]).cuda()
        x_u_s = torch.stack([strong_aug(img) for img in x_u_bg_removed]).cuda()

        loss = fixmatch_loss(model, x_l_aug, y_l, x_u_w, x_u_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
