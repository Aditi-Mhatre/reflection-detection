import torch 
import torchvision
from inspection_dataset import InspectDataset
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, Dice, JaccardIndex
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def save_checkpoint(state, filename="inspect-attentionunet.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def read_path_from_txt(file_path):
    with open(file_path, "r") as f:
        paths = [line.strip() for line in f.readlines()]
    return paths 

def get_loaders(
        train_dir, 
        train_mask_dir, 
        batch_size, 
        train_transform, 
        val_transforms, 
        pin_memory = True,
):
    generator = torch.Generator().manual_seed(42)
    ds = InspectDataset(
        image_dir= train_dir,
        mask_dir = train_mask_dir,
    )

    train_ind, val_ind = train_test_split(list(range(len(ds))), test_size=0.2, random_state=42)
    
    train_ds = InspectDataset(image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform)
    val_ds = InspectDataset(image_dir=train_dir, mask_dir=train_mask_dir, transform=val_transforms)
    train_ds = torch.utils.data.Subset(train_ds, train_ind)
    val_ds = torch.utils.data.Subset(val_ds, val_ind)


    print(f"Train DS size:{len(train_ds)}, Val DS size:{len(val_ds)}  ")

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


def check_iou(preds, labels):
    intersection = torch.logical_and(preds, labels).sum()
    union = torch.logical_or(preds, labels).sum()
    #print(f"inter{intersection}, union{union}  ")
    iou = intersection / (union + 1e-8)
    return iou

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
            preds = torch.sigmoid(predictions)
            preds = (preds > 0.5).float()
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




