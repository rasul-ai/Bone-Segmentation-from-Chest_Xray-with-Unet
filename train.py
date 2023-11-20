#########################################################################################
###This code is perfectly worked with GPU.Solving issues of scaler and torch.cuda.amp###
#########################################################################################
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import Unet
from utils import *

lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
num_epochs = 5
num_workers = 1
image_height = 512
image_width = 512
pin_memory = True
load_model = False

train_img_dir = "./data/10img_all/train/imgs"
train_mask_dir = "./data/10img_all/train/masks"
val_img_dir = "./data/10img_all/val/imgs"
val_mask_dir = "./data/10img_all/val/masks"

def train(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    model.train()  # Set the model to training mode

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            # A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2() 
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = Unet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, val_loader = get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform
    )
    # Check if a checkpoint exists to resume training
    if os.path.exists("checkpoint.pth.tar"):
        checkpoint = torch.load("checkpoint.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch after the loaded checkpoint
        print("Resuming training from epoch", start_epoch)
    else:
        start_epoch = 1

    for epoch in range(start_epoch,num_epochs+1):
        print(f'###Epoch:{epoch}/{num_epochs}')
        train(train_loader, model, optimizer, loss_fn) 

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,  # Include the training loss in the checkpoint
            "val_loss": val_loss,      # Include the validation loss in the checkpoint
            }

        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=device)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_img/", device=device
        )
        
if __name__ == "__main__":
    main()
