import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/drones/train"
VAL_DIR = "data/drones/val"
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 5
VAL_BATCH_SIZE = 2
NUM_WORKERS = 2
IMAGE_SIZE = 300
CHANNELS_IMG = 3
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_UNET = "UNET.pth.tar"
GRADIENT_CLIPPING = 1.0
AMP = True
MODEL_PATH = "models_saved/UNET.pth.tar"
SAVE_EVERY = 5

train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],        
            additional_targets={"image0": "image"}
    )

val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],  
            additional_targets={"image0": "image"}
    )