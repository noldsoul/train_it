import sys 
import os
path  = os.getcwd()
sys.path.append(path)

import torch
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary
from PIL import Image
from tqdm import tqdm, tnrange, tqdm_notebook
import math
from collections import defaultdict
from dataloaders import FluxData, BDD_Data
from models.fcn_hardnet import hardnet
from RAdam import RAdam
from LookAhead import Lookahead
from earlystopping import EarlyStopping
import wandb
wandb.init(project = " BDD_Data 1000 epochs")

### Checking for GPU-----------------------------------------------------------------------------------------
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')


SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch




def cyclical_lr(stepsize, min_lr=3e-2, max_lr=3e-4):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

### Preprocessing--------------------------------------------------------------------------------------
transform = transforms.Compose(transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)))
# dataset = FluxData('/home/chandradeep_p/Desktop/datasets/Flux_data/dataset2.0', transform = None)
dataset = BDD_Data('/content/work/Segmentation/bdd100k', transform = None)
print(len(dataset))
# dataset = BDD_Data('/home/deeplearning/Desktop/chandradeep/dataset/bdd1000/train', transform = transform)
trainset, valset = random_split(dataset,[6000, 1000])
train_loader = DataLoader(trainset, batch_size = 16, shuffle = True )
val_loader = DataLoader(valset, batch_size =16, shuffle = False)


model = hardnet(n_classes=5)
wandb.watch(model, log = 'all')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.)
step_size = 4*len(train_loader)
clr = cyclical_lr(step_size)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])
# optimizer = RAdam( model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, degenerated_to_sgd=True)
# optimizer =  Lookahead(base_optimizer,1e-3 ,k = 6)
# optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.99)
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)
# def show_batch():
#     image, mask = next(iter(train_loader))
#     image = image[:,:,:,:].permute([0,2,3,1]).numpy()
#     for batch_size in range(16):
#         plt.imshow(image[batch_size, :,:,:] / 255)
#         plt.imshow(mask[batch_size, :,:])
#         plt.show()




### Training the model--------------------------------------------------------------------------------
n_epochs = 1000
patience = 20
train_losses = []
val_losses = []
avg_train_losses = []
avg_val_losses = []
avg_threshold =[]
early_stopping = EarlyStopping(patience=patience, verbose=True, delta = 0.005, diff =0.05)
# valid_loss_min = np.Inf
epoch_tqdm = tqdm(total = n_epochs, desc = 'epochs')
for epoch in range(n_epochs):
    train_tqdm = tqdm(total = len(train_loader), desc = 'training batch')
    ###################
    # train the model #
    ###################
    model.train()
    for batch_idx, (image, mask) in enumerate(train_loader):
        if train_on_gpu:
            image, mask = image.cuda(), mask.cuda()
            model = model.cuda()
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        loss.backward()
        scheduler.step() #scheduler step
        optimizer.step()
        train_losses.append(loss.item())
        train_tqdm.update(1)
    train_tqdm.close()
    ######################    
    # validate the model #
    ######################
    valid_tqdm = tqdm(total = len(val_loader), desc ='validation batch')
    model.eval()
    threshold = []
    for batch_idx,(image, mask) in enumerate(val_loader):
        if train_on_gpu:
            image, mask = image.cuda(), mask.cuda()
        output = model(image)
        loss = criterion(output, mask)
        val_losses.append(loss.item())
        output_iou = torch.argmax(output, dim = 1)
        threshold.append(iou_pytorch(output_iou, mask))
        valid_tqdm.update(1)
    valid_tqdm.close()

    # calculate average losses
    train_loss = np.average(train_losses)
    val_loss = np.average(val_losses)
    avg_train_losses.append(train_loss)
    avg_val_losses.append(val_loss)
    avg_threshold.append(torch.mean(torch.stack(threshold)))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tThreshold:{:.6f}'.format(
        epoch+1, train_loss, val_loss, torch.mean(torch.stack(threshold))))
    train_losses = []
    val_losses = []
    early_stopping(val_loss, model, train_loss)
    epoch_tqdm.update(1)
    wandb.log({"Training Loss per epoch": train_loss, "validation Loss per epoch": val_loss})
    wandb.log({"IOU": torch.mean(torch.stack((threshold)))})
    wandb.log({'learning rate': get_lr()})
    if early_stopping.early_stop:
      print('Early Stopping')
      break
