import torch
import torch.nn.functional as F

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
    d_loss = dice_loss(x, y)
    print(d_loss)