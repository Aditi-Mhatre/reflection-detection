import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, JaccardIndex, Dice
from model import UNet, AttentionUNet
import torchvision
import logging
from inspect_utils import(
    load_checkpoint,
    save_checkpoint, 
    get_loaders,
    check_accuracy,
    check_iou,
    #save_predictions_as_imgs,

)
import wandb


torch.cuda.empty_cache()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 16
# EPOCHS = 10
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
MODEL_SAVE_PATH = "./models/finetuned_shiq_unet.pth"
TRAIN_IMG_DIR = "./inspection-data/train/images/"
TRAIN_MASK_DIR = "./inspection-data/train/masks/"
LOG_FILE = "./inspection-data/training_new.log"



def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    epoch_loss = 0
    for batch_idx, img_mask in enumerate(loop):
        img = img_mask[0].float().to(device=DEVICE)
        mask = img_mask[1].float().unsqueeze(1).to(device=DEVICE)
        y_pred = model(img)
        #plt.imsave('pred_8.png', y_pred.squeeze().detach().cpu(), cmap="gray")
        optimizer.zero_grad()
        loss = loss_fn(y_pred, mask)
        epoch_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    
    # with torch.cuda.amp.autocast():
    #     predictions = model(data)
    #     loss = loss_fn(predictions, targets)
    
    # optimizer.zero_grad()
    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()

    #loop.set_postfix(loss=loss.item())
    return epoch_loss/len(loader)


def evaluate_fn(loader, model, loss_fn, device):
    #model.eval()
    #accuracy_metric = Accuracy(task='binary', num_classes=1).to(device)
    #iou_metric = JaccardIndex(task='binary',num_classes=1).to(device)
    epoch_loss = 0
    with torch.no_grad():
        for batch_idx, img_mask in enumerate(tqdm(loader)):
            img = img_mask[0].float().to(device=DEVICE)
            mask = img_mask[1].float().unsqueeze(1).to(device=DEVICE)
            y_pred = model(img)
            loss = loss_fn(y_pred, mask)
            #plt.imsave('evaluation_8.png', y_pred.squeeze().detach().cpu(), cmap="gray")
            # preds = preds.to(device=device)
            # y = y.float().unsqueeze(1).to(device=device)
            epoch_loss += loss.item()
            #accuracy_metric.update(y_pred, mask.int())
            #iou_metric.update(y_pred, mask.int())
        #accuracy = accuracy_metric.compute().item()
        #iou = iou_metric.compute().item()
    return epoch_loss / len(loader)

def freeze_layers(model):
    for param in model.downs.parameters():
        param.requires_grad = False
    for param in model.bottleneck.parameters():
        param.requires_grad = False
    for param in model.final_conv.parameters():
        param.requires_grad = False

def unfreeze_layers(model):
    for param in model.ups.parameters():
        param.requires_grad = True

def main():
    run = wandb.init(project="TL New SHIQ UNet Experiments + CE", config={"learning_rate": 1e-3, "epochs": 200, "batch_size": 16} )
    LEARNING_RATE = wandb.config.learning_rate
    BATCH_SIZE = wandb.config.batch_size
    EPOCHS = wandb.config.epochs
    

    train_transform = A.Compose(
        [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0,0.0,0.0],
            std=[1.0,1.0,1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
        ], is_check_shapes=False
    )

    val_transforms = A.Compose(
        [
        A.Resize(height= IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0,0.0,0.0],
            std=[1.0,1.0,1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),

        ], is_check_shapes=False
    )

    train_loaders, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        PIN_MEMORY,
    )
    features = [64, 128, 256, 512]
    model = UNet(in_channels=3, out_channels=1, features=features).to(DEVICE)
    checkpoint = torch.load("./models/shiq-unet-sweep.pth")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False )

    # for param in model.parameters():
    #     param.requires_grad = False
    freeze_layers(model)
    # for param in model.final_conv.parameters():
    #     param.requires_grad = True
    unfreeze_layers(model)

    #model.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    # print(f"Total Parameters:{total_params}" )
    # print(f"Total FLOPs:{flops.total() / 1e9} GFLOPs")

    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("new_inspect_pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(EPOCHS):
        #training
        logging.info(f"Starting epoch {epoch+1} ")
        train_loss = train_fn(train_loaders, model, optimizer, loss_fn, scaler)

        #validation
        model.eval()
        val_loss = evaluate_fn(val_loader, model, loss_fn, DEVICE)
        val_accuracy, val_dice, val_iou = check_accuracy(val_loader, model, device=DEVICE)
        #logging.info(f"Train Loss:{train_loss: .4f}, Validation Loss: {val_loss:.4f}', Validation Accuracy:{val_accuracy: .4f}, IOU score:{val_iou: .4f}, Dice:{val_dice: .4f}  ")
        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss, 
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy, 
            "IOU score": val_iou, 
            "Dice": val_dice}
            , step=epoch)


        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        logging.info(f"Saved checkpoint for epoch {epoch + 1}  ")
        save_checkpoint(checkpoint)
        check_accuracy(val_loader, model, device=DEVICE)
        logging.info(f"Finished epoch{epoch+1} ")
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()