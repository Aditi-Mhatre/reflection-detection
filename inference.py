import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from PIL import Image
from inspection_dataset import InspectDataset
from model import UNet, AttentionUNet
from unetr_new import UNetR2D
from unetr_af import UNetR2D_AF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import check_iou, FocalTverskyLoss
import segmentation_models_pytorch as smp

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

""" 
    Evaluates Model on a Test Dataset 
    returns accuracy, mean IoU and Dice score
"""
def evaluate_test(
        data_path, data_path_mask, model, folder="./inspection-data/test/images/", DEVICE="cuda"
):

    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    test_ds = InspectDataset(
        image_dir= data_path,
        mask_dir = data_path_mask,
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False
    )

    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0

    with torch.no_grad():
        for batch_idx, img_mask in enumerate(tqdm(loader)):
            img = img_mask[0].float().to(device=DEVICE)
            mask = img_mask[1].float().unsqueeze(1).to(device=DEVICE)
            pred = torch.sigmoid(model(img))
            pred = (pred > 0.5).float()
            iou_score += check_iou(pred, mask)
            dice_score += (2 * (pred * mask).sum())/((pred + mask).sum() + 1e-8)
            num_correct += (pred == mask).sum()
            num_pixels += torch.numel(pred)
            """ Save predictions from the model in a folder -> change directory path"""
            #torchvision.utils.save_image(pred, f"{folder}/pred_{batch_idx+1}.png")
    accuracy = num_correct/num_pixels*100
    avg_dice = dice_score/len(loader)
    avg_iou = iou_score/len(loader)
    return accuracy, avg_iou, avg_dice



""" Draws Contours and Centroids based on the values returned"""
def drawContoursandCentroids(image, contours, centroids):
    img_cv = np.array(image)[:, :, ::-1].copy()
    cv.drawContours(img_cv, contours, -1, (255,0,0), 2)
    for centroid in centroids:
        cv.circle(img_cv, centroid, 5, (0,0,255), -1)
    return img_cv



""" Infer model on a single image """
def single_image_inference(image_pth, mask_pth, model, device):
    model.eval()
    
    iou_scores = []
    dice_scores = []

    # Define transformation (match validation transformations)
    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # Load and preprocess the image
    original_image = Image.open(image_pth).convert('RGB')
    img = transform(image=np.array(original_image))['image'].unsqueeze(0).to(device)  # Apply the transformation
    mask = np.array(Image.open(mask_pth).convert("L"), dtype=np.float32)
    mask[mask == 255.] = 1.0
    augmented_mask = transform(image=mask)
    mask = augmented_mask["image"] 
    mask = mask.to(device)
    # Perform inference
    with torch.no_grad():
        pred = model(img)

    #Post Processing
    # calculate IOU and Dice score
    pred_np = (pred > 0.5).float()
    mask_bin = (mask > 0.5).float()
    iou_value = iou(pred_np, mask_bin)
    dice_value = dice_score(pred_np, mask_bin)
    # Store the scores
    iou_scores.append(iou_value.item())
    dice_scores.append(dice_value.item())
    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)
    print(f"IoU: {avg_iou:.4f}, Dice Score: {avg_dice:.4f}")

    # find contours and centroids
    pred_np = (pred_np.squeeze().cpu().numpy() * 255).astype(np.uint8)
    pred_mask = pred.squeeze(0).cpu().sigmoid().numpy()

    img = img.squeeze(0).cpu().permute(1, 2, 0).numpy()

    contours, centroids = findCentroid(pred)
    original_width, original_height = original_image.size
    scaleX = original_width/IMAGE_WIDTH
    scaleY = original_height/IMAGE_HEIGHT

    # Adjust contours and centroids to match original image size
    original_contours = [contour * np.array([[scaleX, scaleY]]) for contour in contours]
    original_contours = [contour.astype(int) for contour in original_contours]
    original_centroids = [(int(cX * scaleX), int(cY * scaleY)) for cX, cY in centroids]
    print(f"Centroids: {centroids}")
    print(f"Centroids: {original_centroids}")
    img_infer = drawContoursandCentroids(img, contours, centroids)
    org_contour = drawContoursandCentroids(original_image, original_contours, original_centroids)
    fig = plt.figure()
    fig.add_subplot(1, 4, 1)
    plt.imshow(img)
    plt.title('Input Image')

    fig.add_subplot(1, 4, 2)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.imsave("0903_whu_attentionunet_ce.png", pred_mask.squeeze(), cmap='gray')

    fig.add_subplot(1,4,3)
    plt.imshow(img_infer)
    plt.title('Centroids')

    fig.add_subplot(1,4,4)
    plt.imshow(org_contour)
    plt.title("Centroid on original image size")

    iou_value = f"IOU Score: {avg_iou:.4f}"
    dice_value = f"Dice Score: {avg_dice:.4f}"
    plt.text(0.5, 0.85, iou_value, fontsize=12, ha='center', transform=fig.transFigure)
    plt.text(0.5, 0.80, dice_value, fontsize=12, ha='center', transform=fig.transFigure)

    fig.suptitle("Results")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()



def iou(pred, target):
    # Intersection and union
    intersection = (pred * target).sum((1, 2))  
    union = (pred + target - pred * target).sum((1, 2))  
    
    iou = (intersection + 1e-8) / (union + 1e-8)  
    return iou.mean()

def dice_score(pred, target):
    # Dice score calculation
    intersection = (pred * target).sum((1, 2))
    dice = (2 * intersection + 1e-8) / (pred.sum((1, 2)) + target.sum((1, 2)) + 1e-8)
    return dice.mean()


""" Finds centroid from the segmentation mask """
def findCentroid(predicted_mask):
    predicted_mask = (predicted_mask > 0.5).float()  # Ensure mask is binary

    # Convert the mask for OpenCV (uint8)
    mask_np = (predicted_mask.squeeze().cpu().numpy() * 255).astype(np.uint8) 
    contours, _ = cv.findContours(mask_np, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centroids = []

    # calculates the area within of the contours
    for contour in contours:
        M = cv.moments(contour)
        area = cv.contourArea(contour)
        # finds centroids for area over 20
        if area > 20:
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
            else:
                centroids.append((0, 0))
    
    return contours, centroids


if __name__ == "__main__":
    SINGLE_IMG_PATH = "./inspection-data/test/images/image_0903.png"
    SINGLE_MASK_PATH = "./inspection-data/test/masks/mask_0903.png"
    DATA_PATH = "./inspection-data/test/images/"
    DATA_PATH_MASK = "./inspection-data/test/masks/"

    #Change model path 
    MODEL_PATH = "./models/inspect-unetplus_resnet50.pth"

    """ Two Loss function are used: uncomment the one needed
        BCEWithLogitsLoss for CNN-based models (UNet, AttentionUNet, Transfer Learning models, UNet++)
        Focal Tversky Loss for Transformer-based models (UNETR, UNETR-AF) """

    loss_fn = torch.nn.BCEWithLogitsLoss()  
    #loss = FocalTverskyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"


    """ Define Model - should be same as the model saved in the MODEL_PATH
        There are five models : UNet, AttentionUNet, UNet++, UNETR and UNETR-AF

        For UNETR, UNETR_AF = pass the vit configuration in the function
            1. model = UNETR(vit).to(device)  
            2. model = UNETR_AF(vit).to(device)

        For UNet, AttentionUNet, define the model as follows:
            1. model = UNet(in_channels=3, out_channels=1).to(device)
            2. model = AttentionUNet(in_channels=3, out_channels=1).to(device)
        
        For UNet++, make sure that the segmentation_models_pytorch libary is installed and imported:
            import segmentation_models_pytorch as smp

            model = smp.UnetPlusPlus(
                    encoder_name="resnet50",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                ).to(device)

    """

    #Change this line based on the Model you want to use
    model = smp.UnetPlusPlus(
                    encoder_name="resnet50",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                ).to(device)


    #Define configuration for transformer-based models (UNETR, UNETR-AF)

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

    #Loading model 
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # Infer model on a Single Image
    single_image_inference(SINGLE_IMG_PATH, SINGLE_MASK_PATH, model, device)

    # Uncomment this line to - Evaluate Model on a Test Dataset: returns accuracy, IoU score and Dice score 
    # accuracy, iou, dice = evaluate_test(DATA_PATH, DATA_PATH_MASK, model)
    # print(f"Accuracy:{accuracy}, IOU:{iou}, Dice:{dice}")
