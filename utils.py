import torch 
import torchvision
from whu_dataset import WHUDataset
from shiq_dataset import SHIQDataset
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, Dice, JaccardIndex
import numpy as np
from matplotlib import pyplot as plt


def save_checkpoint(state, filename="whu.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


# Adds data augmentation to Train dataset, and retrieves images from the dataset in batches defined
def get_loaders(
        train_dir, 
        train_mask_dir, 
        val_dir, 
        val_mask_dir,
        batch_size, 
        train_transform, 
        val_transforms, 
        pin_memory = True,
        patch_size = None,
        flatten_shape = None
):
    #generator = torch.Generator().manual_seed(42)


    train_ds = WHUDataset(
        image_dir= train_dir,
        mask_dir = train_mask_dir,
        transform=train_transform,
        patch_size= patch_size,
        flatten_shape=flatten_shape,
    )

    val_ds = WHUDataset(
        image_dir= val_dir,
        mask_dir = val_mask_dir,
        transform=val_transforms,
        patch_size=patch_size,
        flatten_shape=flatten_shape,
    )


    #train_ds, val_ds = random_split(train_ds, [0.8,0.2], generator=generator)

    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=True,
    )


    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle = False,
    )

    return train_loader, val_loader


# Calculate IoU
def check_iou(preds, labels):
    intersection = torch.logical_and(preds, labels).sum()
    union = torch.logical_or(preds, labels).sum()
    #print(f"inter{intersection}, union{union}  ")
    iou = intersection / (union + 1e-8)
    return iou


# Function to mesaure the accuracy, IoU, and dice scores
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    model.eval()
    

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            predictions = model(x)
            #plt.imsave("predictions_before_sig_9.png", predictions.squeeze().detach().cpu(), cmap="gray")
            preds = torch.sigmoid(predictions)
            #plt.imsave("predictions_after_9.png", preds.squeeze().detach().cpu(), cmap="gray")
            preds = (preds > 0.5).float()
            #plt.imsave("y_9.png", y.squeeze().detach().cpu(), cmap="gray")
            #plt.imsave("preds_after_threshold_9.png", preds.squeeze().detach().cpu(), cmap="gray")
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum())/((preds + y).sum() + 1e-8)
            iou_score += check_iou(preds,y)
       
    
    accuracy = num_correct/num_pixels*100
    avg_dice = dice_score/len(loader)
    avg_iou = iou_score/len(loader)
    print(f"Got {num_correct} / {num_pixels} with acc {num_correct/num_pixels*100: .2f}")

    model.train()
    return accuracy, avg_dice, avg_iou


# Definge Dice Cross Entropy loss -- not used 
class DiceCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(DiceCrossEntropyLoss, self).__init__()
    
    def forward(self, predictions, targets):
        predictions = F.softmax(predictions, dim=1)
        num_classes = predictions.size(1)
        predictions = predictions.view(predictions.size(0), num_classes, -1)
        targets = targets.view(targets.size(0), num_classes, -1)

        dice_loss = 0.0
        for c in range(num_classes):
            pred_flat = predictions[:, c, :] 
            target_flat = targets[:, c, :]
            intersect = (pred_flat * target_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
            dice_score = 2.0 * intersect / (union + 1e-8)
            dice_loss += (1.0 - dice_score.mean()) 
        dice_loss /= num_classes

        log_pred = torch.log(predictions + 1e-8)
        cross_entropy_loss = -torch.mean(torch.sum(targets * log_pred, dim=1))
        total_loss = dice_loss + cross_entropy_loss

        return total_loss


# Define Focal Tversky Loss
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-8):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        
        # Calculate True Positives, False Negatives and False Positives
        true_pos = torch.sum(y_true_flat * y_pred_flat)
        false_neg = torch.sum(y_true_flat * (1 - y_pred_flat))
        false_pos = torch.sum((1 - y_true_flat) * y_pred_flat)
        
        # Calculate the Tversky index
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        
        focal_tversky_loss = torch.pow((1 - tversky_index), self.gamma)

        return focal_tversky_loss