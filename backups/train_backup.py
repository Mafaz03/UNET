# Gradient problem, gets weird, precition becomes nan
# Backup file in case I need to copy some code
# Do not use and expect good result

import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import ImageDataset
import config
from torch import optim
from .. import unet
from torch import nn
from dice_score import dice_loss
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Creating Dataloader
train_set = ImageDataset(config.TRAIN_DIR)
train_dl = DataLoader(train_set, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE)

val_set = ImageDataset(config.VAL_DIR)
val_dl = DataLoader(val_set, shuffle=False, batch_size=config.VAL_BATCH_SIZE)

# Model and Loss function
model = unet.UNet(3, 3).to(config.DEVICE)
optimizer = optim.RMSprop(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-8, momentum=0.999, foreach=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
grad_scaler = torch.amp.GradScaler('cuda')
criterion = nn.CrossEntropyLoss()

global_step = 0

for epoch in range(1, config.NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    with tqdm(total=len(train_dl), desc=f'Epoch {epoch}/{config.NUM_EPOCHS}', unit='img') as pbar:
        for batch in train_dl:
            images = images.to(device=config.DEVICE, dtype=torch.float32, memory_format=torch.channels_last)
            targets = targets.to(device=config.DEVICE, dtype=torch.long)

            with torch.autocast(config.DEVICE if config.DEVICE != 'mps' else 'cpu', enabled=True):
                targets_pred = model(images)
                loss = criterion(targets_pred, targets.to(torch.float))
                loss += dice_loss(
                    targets_pred.float(),
                    targets.float(),
                )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIPPING)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()
    print("Epoch loss: ", epoch_loss)