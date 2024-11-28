import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, JaccardIndex, Dice
from model import UNet, AttentionUNet
from fvcore.nn import FlopCountAnalysis
from unetr import UNetR2D
from unetr_af import UNetR2D_AF
import torchvision
import logging
from utils import(
    load_checkpoint,
    save_checkpoint, 
    get_loaders,
    check_accuracy,
    check_iou,
    FocalTverskyLoss,
    #save_predictions_as_imgs,

)

from matplotlib import pyplot as plt
import wandb

#run = wandb.init(project="Inspection AttentionUNet Experiments + FL", config={"learning_rate": 1e-3, "epochs": 50, "batch_size": 16} )



#LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 16
# EPOCHS = 10
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PIN_MEMORY = True
LOAD_MODEL = False
MODEL_SAVE_PATH = "./models/inspect_eca_unetr.pth"
TRAIN_IMG_DIR = "./SHIQ/train/images"
TRAIN_MASK_DIR = "./SHIQ/train/masks"
TEST_IMG_DIR = "./SHIQ/test/images"
TEST_MASK_DIR = "./SHIQ/test/masks"
LOG_FILE = "./SHIQ/training_shiq_unetr-af.log"
PRED_PATH = "./SHIQ/predictions/"


sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_iou"},
    "parameters": {
        "batch_size":{"values":[16] } ,
        "epochs": {"values": [15, 30, 50, 100]},
        "learning_rate": {"values":[1e-4, 3e-3, 1e-3, 3e-2, 1e-2]},
        "optimizer":{"values":["adam", "sgd"]},
    },
}
torch.cuda.empty_cache()

#sweep_id = wandb.sweep(sweep=sweep_configuration, project="SHIQ AttentionUNet Experiments + CE")

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

def save_predictions_as_imgs(
        loader, model, folder="./SHIQ/predictions/", device="cuda"
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            #preds = (preds > 0.5).float()
        # torchvision.utils.save_image(
        #     preds, f"{folder}/pred_{idx}.png"
        # )
        # torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/predictions_{idx}.png")
    model.train()


def main():
    run = wandb.init(project="SHIQ UNetR-AF ECA Experiments + FTL", config={"learning_rate": 1e-3, "epochs": 50, "batch_size": 4} )

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


    model = UNetR2D_AF(vit).to(DEVICE)
    #loss_fn = nn.BCEWithLogitsLoss()
    # if OPTIMIZER == "adam":
    #     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # elif OPTIMIZER== "sgd":
    #     optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    loss_fn = FocalTverskyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # total_params = sum(p.numel() for p in model.parameters())
    # flops = FlopCountAnalysis(model, torch.randn((1,3,200,200)))

    train_loaders, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        PIN_MEMORY,
    )
    # print(f"Total Parameters:{total_params}" )
    # print(f"Total FLOPs:{flops.total() / 1e9} GFLOPs")

    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("new_inspect_pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    best_metric = 0
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

        current_metric = val_iou
        if current_metric < best_metric:
            best_metric = current_metric
            logging.info(f"Finished epoch{epoch+1} ")
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, MODEL_SAVE_PATH)
            

        # checkpoint = {
        #     "state_dict": model.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        # }
        # logging.info(f"Saved checkpoint for epoch {epoch + 1}  ")
        # save_checkpoint(checkpoint)
        # check_accuracy(val_loader, model, device=DEVICE)
        # # save_predictions_as_imgs(
        # #     val_loader, model, folder="./data/predictions/", device=DEVICE
        # # )
        # logging.info(f"Finished epoch{epoch+1} ")
        # torch.save({
        #     'epoch': epoch, 
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': train_loss,
        # }, MODEL_SAVE_PATH)
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

if __name__ == "__main__":
    #wandb.agent(sweep_id, function=main, count=20)
    main()