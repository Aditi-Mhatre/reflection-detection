import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, JaccardIndex, Dice
from model import UNet, AttentionUNet
import wandb
import torchvision
import logging
from utils import(
    load_checkpoint,
    save_checkpoint, 
    get_loaders,
    check_accuracy,
    check_iou,
    #save_predictions_as_imgs,

)
from matplotlib import pyplot as plt


# Initialize experiments for Weights and Biases dashboard
# comment the line based on which model to train
run = wandb.init(project="WHU UNet Experiments", config={"learning_rate": 1e-3, "epochs": 15, "batch_size": 32} )
#run = wandb.init(project="WHU UNet Experiments", config={"learning_rate": 1e-3, "epochs": 15, "batch_size": 32} )


LEARNING_RATE = wandb.config.learning_rate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = wandb.config.batch_size
EPOCHS = wandb.config.epochs
IMAGE_HEIGHT = 200
IMAGE_WIDTH = 200
PIN_MEMORY = True
LOAD_MODEL = False
MODEL_SAVE_PATH = "./models/optimal-whu-unet.pth"
TRAIN_IMG_DIR = "./whu/train/"
TRAIN_MASK_DIR = "./whu/train_masks/"
TEST_IMG_DIR = "./whu/tests/"
TEST_MASK_DIR = "./whu/test_masks/"
LOG_FILE = "./training_whu.log"
PRED_PATH = "./whu/predictions/"

logger = logging.getLogger(__name__)
logging.basicConfig(filename=LOG_FILE, encoding= 'utf-8',level=logging.INFO)
logging.info("started training")


sweep_config ={
    "method": "random",
    "metric":{
        "name": "val_iou",
        "goal": "maximize"
    },  
    "parameters":{
        "learning_rate":{
            "values":[0.1, 1e-2, 1e-3, 1e-4] 
        },
        "batch_size":{
            "values":[16,32] 
        },
        "epochs":{
            "values":[20,50] 
        } 
    } 
}  
 

#sweep_id = wandb.sweep(sweep_config, project="WHU UNet Experiments")

training_loss =[]
validation_loss =[]  

def train_fn(loader, model, optimizer, loss_fn, scaler):
    wandb.watch(model,loss_fn, log="all", log_freq=10)
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
        #wandb.log({"batch loss": loss.item()} )

    
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
        loader, model, folder="./whu/predictions/", device="cuda"
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


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    return optimizer



def main():
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

    # Initialize the model - 
    #UNet
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    #Attention UNet
    #model = AttentionUNet(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #optimizer = build_optimizer(model, config.optimizer, config.learning_rate)

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

    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("new_inspect_pth.tar"), model)
    
    
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(wandb.config.epochs):
        #training
        logging.info(f"Starting epoch {epoch+1} ")
        train_loss = train_fn(train_loaders, model, optimizer, loss_fn, scaler)

        #validation
        model.eval()
        val_loss = evaluate_fn(val_loader, model, loss_fn, DEVICE)
        val_accuracy, val_dice, val_iou = check_accuracy(val_loader, model, device=DEVICE)
        logging.info(f"Train Loss:{train_loss: .4f}, Validation Loss: {val_loss:.4f}', Validation Accuracy:{val_accuracy: .4f}, IOU score:{val_iou: .4f}, Dice:{val_dice: .4f}  ")
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
        # save_predictions_as_imgs(
        #     val_loader, model, folder="./whu/predictions/", device=DEVICE
        # )
        logging.info(f"Finished epoch{epoch+1} ")
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, MODEL_SAVE_PATH)

if __name__ == "__main__":
    #wandb.agent(sweep_id, main, count=5)
    main()
    
