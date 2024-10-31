import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import config
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        super().__init__()

        self.root_dir = root_dir
        self.list_files = os.listdir(root_dir)
        self.transforms = transforms
        # print(self.list_files)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        image_path = self.list_files[index]
        file_path = os.path.join(self.root_dir, image_path)
        image = np.array(Image.open(file_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]
        
        # import pdb; pdb.set_trace()
        if self.transforms:
            augmentations = self.transforms(image=input_image, image0=target_image)
            input_image = augmentations["image"]
            target_image = augmentations["image0"]


        return input_image, target_image
    
if __name__ == "__main__":
    train_set = ImageDataset(config.TRAIN_DIR, transforms=config.train_transform)
    train_dl = DataLoader(train_set, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE)
    next(iter(train_dl))