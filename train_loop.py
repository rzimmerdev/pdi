# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader, random_split
from unet import UNet

import os

PROCESSED_IMAGES_PATH = 'data/processed'
ORIGINAL_IMAGES_PATH = 'data/original'
"""# Dataset creation"""
def get_data(path):
    ''''
    Creation of a list with all paths to the images and their masks
        Args:
            path (str): string with the path to the images
        Returns:
            paths (list): list with all paths
    '''
    paths = []
    tiles = os.listdir(path)
    for tile in range(1, len(tiles)):
        images = os.listdir(f"{path}/{tile}/images")
        for image in images:
            paths.append({"image": f"{path}/{tile}/images/{image}", "label": f"{path}/{tile}/masks/{image}"})
    return paths


class ProcessedImagesDataset(Dataset):
    ''''
    Creation of dataset after list creation 
    Reading grayscale image
    '''
    def __init__(self, img_dir):
        self.data = get_data(img_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["image"]
        mask_path = self.data[idx]["label"]
        image, mask = read_image(image_path), read_image(mask_path, ImageReadMode.GRAY)

        return image, mask

"""# Treino"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_accuracy(loader, model, device):
    ''''
    Check accuracy and dice score in each train loop
        Args:
            loader (torch.utils.data.ImageDataLoader) : Data Loader that loads the images
            model (torch.nn.Module): The model to have its accuracy be evaluated
            device (torch.device): The chosen hardware device

    '''
    num_correct = 0
    num_pixels = 0
    model.eval()
    dice_score = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # Label doesn't have a channel

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum())/((preds + y).sum() + 1e-8)

    print(f"Dice score: {dice_score/ len(loader)}")
    print(f"Got {num_correct}/{num_pixels} with accuracy {num_correct*100/num_pixels:.2f}")

#pip install UNet

from unet import UNet
import torch.optim as optim
def train_fn(loader: DataLoader, model: torch.nn.Module, optimizer: optim.Optimizer, loss_fn, scaler, epochs):

    ''''
    Training functionality based on the chosen parameters.
        Args:
        loader (torch.utils.data.ImageDataLoader): Data Loader that loads the images
        model (torch.nn.Module) : The model to have its accuracy be evaluated
        optimizer (optim.Optimizer): The chosen optimizer for the training of the model
        loss_fn (torch.cuda.amp.GradScaler()): The chosen loss function for the training of the model
        scaler (torch.cuda.amp.GradScaler()): The chosen scaler for the data
        epochs (int) : The number of epochs
    '''
    for epoch in range(epochs):
        model.train()

        batch_loss = 0

        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            assert data.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {data.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

            # forward
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_loss += loss.item()
        print (f'Epoch [{epoch+1}/{epochs}], Loss: {batch_loss:.4f}')

#cd pdi-2022

"""# Main"""

''''
Na função principal, instanciamos o modelo, o dataset, o dataLoader e os parâmetros para o treino,
realizado com cerca de 70% dos dados.
'''
def main():
    torch.cuda.empty_cache()
    model = UNet(3, 1).to(device=device)
    dataset = ProcessedImagesDataset(PROCESSED_IMAGES_PATH)
    train_data, test_data = random_split(dataset, [50, 22])
    dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
    loss_fn = nn.BCEWithLogitsLoss()
    grad_scaler = torch.cuda.amp.GradScaler()
    epochs = 10
    try:
        train_fn(dataloader, model, optimizer, loss_fn, grad_scaler, epochs)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
    torch.save(model.state_dict(), 'model.pth')
    return model
predictor = main()

ds = ProcessedImagesDataset(PROCESSED_IMAGES_PATH)
print(ds[0][0].shape)

image = torch.reshape(ds[0][0], (1, 3, 256, 256)).to(device, dtype=torch.float32)
image.shape

import matplotlib.pyplot as plt
import numpy as np

''''
Predições realizadas com a rede
'''
with torch.no_grad():
    predictor = UNet(3, 1).cuda()
    predictor.load_state_dict(torch.load("model.pth"))
    predictor.eval()
    predictions = predictor(image)[0].cpu()
    predictions = np.transpose(predictions, axes=[1, 2, 0])
    predictions = np.array(predictions.reshape(predictions.shape[0], predictions.shape[1]))
    predictions = (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))
    plt.imshow(predictions, cmap="gray")

