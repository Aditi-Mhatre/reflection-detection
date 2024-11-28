import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True #resolves ValueError "Decompressed data too large"


class WHUDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        # Display the image
        # plt.imshow(image)
        # plt.axis('off')  # Hide axes
        # plt.show()

        # # Display the mask
        # plt.imshow(mask, cmap='gray')  # Use 'gray' colormap for single-channel images
        # plt.axis('off')  # Hide axes
        # plt.show()

        # # Save the image
        # plt.imsave('image_5.png', image)
        # # Save the mask
        # plt.imsave('mask_5.png', mask, cmap='gray')
            
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask