import torch
import os
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from torchmetrics import F1Score


def compute_f1(model, loader):
    model = model.eval().to(config.DEVICE)
    f1 = F1Score(task="multiclass", num_classes=config.NUM_CLASSES).to(config.DEVICE)
    for idx, (features, labels) in enumerate(tqdm(loader)):
        features = features.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        with torch.no_grad():
            logits = model(features)
        preds = torch.argmax(logits, dim=1)
        f1(preds, labels)
    return f1


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
