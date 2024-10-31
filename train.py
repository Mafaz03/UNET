import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import ImageDataset
import config
from torch import optim
from unet import UNET
from torch import nn
from dice_score import dice_loss
import torch.nn.functional as F
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Creating Dataloader
train_set = ImageDataset(config.TRAIN_DIR, transforms=config.train_transform)
train_dl = DataLoader(train_set, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE)

val_set = ImageDataset(config.VAL_DIR, transforms=config.val_transforms)
val_dl = DataLoader(val_set, shuffle=False, batch_size=config.VAL_BATCH_SIZE)


model = UNET(in_channels=3, out_channels=3).to(config.DEVICE)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scaler = torch.amp.GradScaler('cuda')

if config.LOAD_MODEL:
    config.load_checkpoint(torch.load(config.MODEL_PATH), model)

for epoch in range(config.NUM_EPOCHS):
    dice_score = 0
    losses = 0
    num_batches = 0
    for images, targets in train_dl:
        images, targets = images.to(config.DEVICE), targets.float().to(config.DEVICE)
        predictions = model(images)
        utils.plot_batch(torch.sigmoid(predictions), save_pth=f"evaluation/{epoch}.jpg", show=False)
        # utils.plot_batch(predictions, save_pth=f"evaluation2/{epoch}.jpg", show=False)

        with torch.amp.autocast('cuda'):
            predictions = model(images)
            loss = loss_fn(predictions, targets)
            losses += loss
        
        dice_loss = utils.dice_loss(torch.sigmoid(predictions), targets)
        dice_score += (1 - dice_loss.item()) * 100  # Accumulate score
        num_batches += 1 

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    avg_loss = losses / num_batches
    avg_dice_score = dice_score / num_batches

    print("Epoch: ", epoch, "| Average loss: ", round(avg_loss.item(), 3), "| Average Dice Score: ", round(avg_dice_score, 3), " %")

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
        }
    if epoch % config.SAVE_EVERY == 0 or epoch == config.NUM_EPOCHS-1:
        utils.save_checkpoint(checkpoint)