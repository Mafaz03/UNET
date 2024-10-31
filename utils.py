import matplotlib.pyplot as plt
import os
import config
import torch

def plot_images(images, num_images=4, save_pth=None, show=False, **kwargs):
    """
    Plot a specified number of images and their corresponding targets.

    Parameters:
    - images: Tensor of images.
    - num_images: Number of images to display.
    """
    num_images = min(num_images, len(images))  # Ensure we don't exceed available images
    plt.figure(**kwargs)
    
    for i in range(num_images):
        # Plot the input image
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].to("cpu").permute(1, 2, 0).detach().numpy())
        plt.axis('off')

    if save_pth:
        if "/" in save_pth:
            os.makedirs('/'.join(save_pth.split('/')[:-1]), exist_ok=True)
        plt.savefig(save_pth, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    if show: plt.show()


def plot_batch(tensor_image, save_pth=None, show=True, **kwargs):
    plt.figure(**kwargs)
    if len(tensor_image.shape) != 4:
        if tensor_image.shape[-1] > tensor_image.shape[-3]: # not so good approach, but it works
            tensor_image = tensor_image.to("cpu").permute(1,2,0).detach().numpy()
        else: tensor_image = tensor_image.to("cpu").detach().numpy()
        plt.imshow(tensor_image)
        if save_pth:
            if "/" in save_pth:
                os.makedirs('/'.join(save_pth.split('/')[:-1]), exist_ok=True)
            plt.savefig(save_pth, bbox_inches='tight', dpi=300)
    else:
        plot_images(tensor_image.clip(0), save_pth=save_pth)
    if show:
        plt.show()

def save_checkpoint(state, filename=config.MODEL_PATH):
    if "/" in filename:
        os.makedirs('/'.join(filename.split('/')[:-1]), exist_ok=True)
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def dice_coefficient(prediction, target, smooth=1e-6):
    """
    Calculate the Dice coefficient.
    :param prediction: Predicted segmentation mask (float tensor)
    :param target: Ground truth segmentation mask (float tensor)
    :param smooth: Smoothing factor to avoid division by zero
    :return: Dice coefficient
    """
    intersection = (prediction * target).sum(dim=[1, 2])  # Sum over H and W
    dice = (2. * intersection + smooth) / (prediction.sum(dim=[1, 2]) + target.sum(dim=[1, 2]) + smooth)
    return dice.mean()  # Average over the batch

def dice_loss(prediction, target):
    """
    Calculate the Dice loss for RGB images.
    :param prediction: Predicted segmentation mask (RGB) [B, C, H, W]
    :param target: Ground truth segmentation mask (RGB) [B, C, H, W]
    :return: Dice loss
    """
    # Convert to binary masks (for segmentation, you might want to threshold)
    prediction = (prediction > 0.5).float()  # Threshold for binary mask
    target = (target > 0.5).float()          # Threshold for binary mask
    
    # Calculate Dice loss for each channel
    dice_r = dice_coefficient(prediction[:, 0:1, :, :], target[:, 0:1, :, :])  # Red channel
    dice_g = dice_coefficient(prediction[:, 1:2, :, :], target[:, 1:2, :, :])  # Green channel
    dice_b = dice_coefficient(prediction[:, 2:3, :, :], target[:, 2:3, :, :])  # Blue channel
    
    # Average Dice loss across channels
    return 1 - (dice_r + dice_g + dice_b) / 3  # Return average Dice loss

if __name__ == "__main__":
    x = torch.rand(2, 3, 512, 512)
    y = torch.rand(2, 3, 512, 512)
    d_loss = dice_loss(x, 1-x)
    print(d_loss)
        