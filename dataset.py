import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from natsort import natsorted


class PacemakerDataset(Dataset):
    def __init__(self,image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        ##################There is a problem here##############
        ##########Images have wrong masks###############
        ###So paths need to be sorted##############
        #####Finally it is solved###############
        self.images = natsorted(os.listdir(image_dir))
        self.masks = natsorted(os.listdir(mask_dir))
        
        # print(self.images)
        # print(self.masks)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir,self.masks[index])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #grayscale

        if self.transform is not None:
            augmentation = self.transform(image = image, mask = mask)
            image = augmentation["image"]
            mask = augmentation["mask"]
        # Convert to float32 data type to solve- 
        # RuntimeError: Input type (unsigned char) and bias type (float) should be the same

        # image = image.astype(np.float32) {astype is not supported in torch.
        #  pytorch have its own data conversion type}
        # mask = mask.astype(np.float32)
        image = image.float()
        mask = mask.float()
        return image, mask