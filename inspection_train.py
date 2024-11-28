import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, JaccardIndex, Dice
from model import UNet, AttentionUNet
from unetr_new import UNetR2D
from unetr_af import UNetR2D_AF
from fvcore.nn import FlopCountAnalysis
import segmentation_models_pytorch as smp
from unetr_new import UNetR2D
import torchvision
import logging

""" 
    Inspect_utils has a different loader than utils, which takes
    only the train directories for images and masks and splits
    them into 80:20 ratio as Inspection Dataset has lesser
    images for training.
    
"""
from inspect_utils import(
    load_checkpoint,
    save_checkpoint, 
    get_loaders,
    check_accuracy,
    check_iou,
    #save_predictions_as_imgs,

)
from utils import FocalTverskyLoss
from matplotlib import pyplot as plt
import wandb

"""
    Initialize project and account for experiments on wandb.ai (Weights ANd Biases)
    wandb.ai is a tool used to monitor the progress the models while training
    i.e wandb.log
    which captures the metrics defined (i.e val_iou, val_acc, dice) and
    parameteres defined in config (i.e learning_rate, epochs, batch_size)

"""

run = wandb.init(project="Inspection AttentionUNet New Experiments + CE", config={"learning_rate": 3e-3, "epochs": 50, "batch_size": 16} )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

#Saves model to this path
MODEL_SAVE_PATH = "./models/inspect-test-new.pth"
TRAIN_IMG_DIR = "./data/image/"
TRAIN_MASK_DIR = "./data/mask/"
LOG_FILE = "./inspection-data/training_new.log"
PRED_PATH = "./inspection-data/predictions/"



sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "iou"},
    "parameters": {
        "batch_size":{"values":[16] } ,
        "epochs": {"values": [15, 30, 50, 100]},
        "learning_rate": {"values":[1e-4, 3e-3, 1e-3, 3e-2, 1e-2]},
        "optimizer":{"values":["adam", "sgd"]},
    },
}

#Empties torch cache 
torch.cuda.empty_cache()


""" Uncomment this line for running sweep """
#sweep_id = wandb.sweep(sweep=sweep_configuration, project="Inspection AttentionUNet New Experiments + CE")

# Configuration for transformer-based models (UNETR, UNETR-AF)
vit = {}
vit["image_size"] = 224
vit["num_layers"] = 12 
vit["hidden_dim"] = 768  
vit["mlp_dim"] = 3072 
vit["num_heads"] = 12
vit["dropout_rate"] = 0.1
vit["patch_size"] = 16
vit["num_patches"] = (vit["image_size"] * vit["image_size"]) // (vit["patch_size"] * vit["patch_size"] )  #img_size * img_size / patch_size * patch_size
vit["num_channels"] = 3
vit["flat_patch_shape"] = (
    vit["num_patches"],
    vit["patch_size"]*vit["patch_size"]*vit["num_channels"]  
) 

logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
logging.info("started training")

def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    epoch_loss = 0
    for batch_idx, img_mask in enumerate(loop):
        img = img_mask[0].float().to(device=DEVICE)
        mask = img_mask[1].float().unsqueeze(1).to(device=DEVICE)
        y_pred = model(img)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, mask)
        epoch_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return epoch_loss/len(loader)


def evaluate_fn(loader, model, loss_fn, device):
    epoch_loss = 0
    with torch.no_grad():
        for batch_idx, img_mask in enumerate(tqdm(loader)):
            img = img_mask[0].float().to(device=DEVICE)
            mask = img_mask[1].float().unsqueeze(1).to(device=DEVICE)
            y_pred = model(img)
            loss = loss_fn(y_pred, mask)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

def save_predictions_as_imgs(
        loader, model, folder="./data/predictions/", device="cuda"
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            #preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/predictions_{idx}.png")
    model.train()


def main():
    LEARNING_RATE = wandb.config.learning_rate
    BATCH_SIZE = wandb.config.batch_size
    EPOCHS = wandb.config.epochs
    #OPTIMIZER = wandb.config.optimizer


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

    """ Define Model - should be same as the model saved in the MODEL_PATH
    There are five models : UNet, AttentionUNet, UNet++, UNETR and UNETR-AF

    For UNETR, UNETR_AF = pass the vit configuration in the function
        1. model = UNETR(vit).to(DEVICE)  
        2. model = UNETR_AF(vit).to(DEVICE)

    For UNet, AttentionUNet, define the model as follows:
        1. model = UNet(in_channels=3, out_channels=1).to(DEVICE)
        2. model = AttentionUNet(in_channels=3, out_channels=1).to(DEVICE)
    
    For UNet++, make sure that the segmentation_models_pytorch libary is installed and imported:
        import segmentation_models_pytorch as smp

        model = smp.UnetPlusPlus(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            ).to(DEVICE)

    """

    #Change this line based on the Model you want to use
    model = AttentionUNet(in_channels=3, out_channels=1).to(DEVICE)
    
    # if OPTIMIZER == "adam":
    #     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # elif OPTIMIZER== "sgd":
    #     optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    """ Select Loss function based on preference"""
    loss_fn = nn.BCEWithLogitsLoss()
    #loss_fn = FocalTverskyLoss()

    """ Select optimizer"""
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    """ Measure models parameters and MACs (multiplication, addition operations)"""
    total_params = sum(p.numel() for p in model.parameters())
    flops = FlopCountAnalysis(model, torch.randn((1,3,200,200)))

    """ Get Loaders from the Dataset """
    train_loaders, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        PIN_MEMORY,
    )

    # print(f"Total Parameters:{total_params}" )
    # print(f"Total FLOPs:{flops.total() / 1e9} MACs")

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
        print(f"Validation Loss:{val_loss} ")
        val_accuracy, val_dice, val_iou = check_accuracy(val_loader, model, device=DEVICE)
        logging.info(f"Train Loss:{train_loss: .4f}, Validation Loss: {val_loss:.4f}', Validation Accuracy:{val_accuracy: .4f}, IOU score:{val_iou: .4f}, Dice:{val_dice: .4f}  ")
        
        """ Log progress into Wandb for Tracking and Monitoring """
        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss, 
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy, 
            "IOU score": val_iou, 
            "Dice": val_dice}
            , step=epoch)
        
        print(f"Train Loss:{train_loss: .4f}, Validation Loss: {val_loss:.4f}', Validation Accuracy:{val_accuracy: .4f}, IOU score:{val_iou: .4f}, Dice:{val_dice: .4f} ")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        logging.info(f"Saved checkpoint for epoch {epoch + 1}  ")
        save_checkpoint(checkpoint)
        check_accuracy(val_loader, model, device=DEVICE)
        # save_predictions_as_imgs(
        #     val_loader, model, folder="./data/predictions/", device=DEVICE
        # )
        logging.info(f"Finished epoch{epoch+1} ")
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, MODEL_SAVE_PATH)

if __name__ == "__main__":
    """ Uncomment this line for running sweep for hyperparameters optimization """
    #wandb.agent(sweep_id, function=main, count=5)

    """ Comment this line if running sweep"""
    main()