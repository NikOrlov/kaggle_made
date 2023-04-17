import os
import torch
import torch.nn.functional as F
import numpy as np
import config
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SportDataset
from efficientnet_pytorch import EfficientNet
from utils import load_checkpoint, save_checkpoint, compute_f1


def save_feature_vectors(model, loader, output_size=(1, 1), file="trainb7", size=1, emb_size=1):
    model.eval()
    images = np.zeros((size, emb_size))
    labels = np.zeros(size)
    bs = config.BATCH_SIZE
    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=output_size).reshape(x.shape[0], -1)
            features = features.cpu().numpy()
        images[idx * bs: (idx + 1) * bs] = features
        labels[idx * bs: (idx + 1) * bs] = y

    np.save(f"data_features/X_{file}.npy", images)
    np.save(f"data_features/y_{file}.npy", labels)
    model.train()


def train_one_epoch(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(config.DEVICE)
        targets = targets.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())


def main():
    model = EfficientNet.from_pretrained("efficientnet-b7")
    emb_size = 2560
    model_type = 'b7'
    model._fc = nn.Linear(emb_size, config.NUM_CLASSES)

    train_dataset = SportDataset(root=config.ROOT, stage='train', transform=config.basic_transform)
    test_dataset = SportDataset(root=config.ROOT, stage='test', transform=config.basic_transform)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    model = model.to(config.DEVICE)

    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, loss_fn, optimizer, scaler)
        f1_score = compute_f1(model, train_loader)
        print(f1_score.compute())
        
    if config.SAVE_MODEL:
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)

    save_feature_vectors(model,
                         train_loader,
                         output_size=(1, 1),
                         file=f"train_{model_type}",
                         size=len(train_dataset),
                         emb_size=emb_size)
    save_feature_vectors(model,
                         test_loader,
                         output_size=(1, 1),
                         file=f"test_{model_type}",
                         size=len(test_dataset),
                         emb_size=emb_size)


if __name__ == "__main__":
    main()
