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
import sys
from torchsummary import summary
import PIL
from tqdm import tqdm, tnrange, tqdm_notebook
from pytorchtools import EarlyStopping
### Checking for GPU-----------------------------------------------------------------------------------------
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

### FCN-Hardnet model----------------------------------------------------------------------------------------
class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias = False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels
 
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          layers_.append(ConvLayer(inch, outch))
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #print("upsample",in_channels, out_channels)

    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True,
                            )
        if concat:                            
          out = torch.cat([out, skip], 1)
          
        return out

class hardnet(nn.Module):
    def __init__(self, n_classes=19):
        super(hardnet, self).__init__()

        first_ch  = [16,24,32,48]
        ch_list = [  64, 96, 160, 224, 320]
        grmul = 1.7
        gr       = [  10,16,18,24,32]
        n_layers = [   4, 4, 8, 8, 8]

        blks = len(n_layers) 
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append (
             ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                       stride=2) )
        self.base.append ( ConvLayer(first_ch[0], first_ch[1],  kernel=3) )
        self.base.append ( ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2) )
        self.base.append ( ConvLayer(first_ch[2], first_ch[3],  kernel=3) )

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append ( blk )
            if i < blks-1:
              self.shortcut_layers.append(len(self.base)-1)

            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1) )
            ch = ch_list[i]
            
            if i < blks-1:            
              self.base.append ( nn.AvgPool2d(kernel_size=2, stride=2) )


        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks-1
        self.n_blocks =  n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])
        
        for i in range(n_blocks-1,-1,-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1))
            cur_channels_count = cur_channels_count//2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])
            
            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels


        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
               padding=0, bias=True)
    def forward(self, x):
        
        skip_connections = []
        size_in = x.size()
        
        
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x
        
        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)
        
        out = self.finalConv(out)
        
        out = F.interpolate(
                            out,
                            size=(size_in[2], size_in[3]),
                            mode="bilinear",
                            align_corners=True)
        return out
        

### Dataset Dataloading -------------------------------------------------------------------------------
class FluxData(Dataset):
    def __init__(self, root_dir, transform = None):
        self.image_path = root_dir + '/images2/'
        self.masks_path = root_dir + '/masks2/'
        self.transform  = transform
        self.to_tensor = transforms.ToTensor()
        self.images = sorted(glob.glob(self.image_path+'*.jpg' ))
        self.masks = sorted(glob.glob(self.masks_path+'*.jpg'))
    def __getitem__(self, idx):
        height = 224
        width = 224
        image = cv2.imread(self.images[idx])
        image = cv2.resize(image, (width, height))
        mask = cv2.imread(self.masks[idx], 0)
        mask = cv2.resize(mask, (width, height))
        class_id = [0, 3, 4, 5,37]
        for i in np.unique(mask):
            if i in class_id:
                mask[mask == i] = class_id.index(i)
            else:
                mask[mask == i] = 0
        seg_labels = torch.from_numpy(mask).long()
        image = torch.from_numpy(image).float().permute([2,0,1])
        return image, seg_labels
    def __len__(self):
        return len(self.images)

### Preprocessing--------------------------------------------------------------------------------------
dataset = FluxData('/home/chandradeep_p/Desktop/datasets/Flux_data/flux_dataset_new', transform = None)
trainset, valset = random_split(dataset,[6000,753])
train_loader = DataLoader(trainset, batch_size = 16, shuffle = True )
val_loader = DataLoader(valset, batch_size =16, shuffle = False)

model = hardnet(n_classes=5)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)

### Learning Rate Scheduler----------------------------------------------------------------------------
optimizer = torch.optim.SGD(model.parameters(), start_lr)
lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (lr_find_epochs * len(train_loader)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
lr_find_loss = []
lr_find_lr = []
iter = 0
smoothing = 0.05
for i in range(lr_find_epochs):
    print("epoch{}".format(i))
    model.train()
    for image, mask in train_loader:
        if train_on_gpu:
            image, mask =image.cuda(), mask.cuda()
            model = model.cuda()
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        scheduler.step()
        lr_step =optimizer.state_dict()['param_groups'][0]['lr']
        lr_find_lr.append(lr_step)

        if iter ==0:
            lr_find_loss.append(loss)
        else:
            loss = smoothing * loss + (1-smoothing) * lr_find_loss[-1]
            lr_find_loss.append(loss)
        iter +=1

### Cyclical Leraning Rate
def cyclical_lr(stepsize, nim_lr =3e-4, max_lr =3e-3):
    scaler = lambda x:1.
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1-x)) * scaler(cycle)
    
    lr_lambda = lambda it:min_lr + (max_lr - min_lr)*relative(it, stepsize)
    return lr_lambda

optimizer = torch.optim.SGD(model.parameters(), lr = 1.)
step_size = 4*len(train_loader)
clr = cyclical_lr(step_size, min_lr = end_lr / factor, max_lr = end_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])


### Training the model--------------------------------------------------------------------------------
n_epochs = 10
valid_loss_min = np.Inf # track change in validation loss
early_stopping = EarlyStopping(patience = patience, verbose = True)
for epoch in tnrange(n_epochs, desc = 'epochs'):
    train_loss = []
    valid_loss = []
    avg_train_loss = []
    avg_valid_loss = []

    outer = tqdm_notebook(total = len(train_loader), desc = 'training batch')
    ###################
    # train the model #
    ###################
    model.train()
    for image, mask in train_loader:
        if train_on_gpu:
            image, mask = image.cuda(), mask.cuda()
            model = model.cuda()
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        outer.update(1)
    outer.close()
    ######################    
    # validate the model #
    ######################
    inner = tqdm_notebook(total = len(val_loader), desc ='validation batch')
    model.eval()
    for image, mask in val_loader:
        if train_on_gpu:
            image, mask = image.cuda(), mask.cuda()
        output = model(image)
        loss = criterion(output, mask)
        valid_loss.append(loss.item())
        inner.update(1)
    inner.close()
    # calculate average losses
    train_losses = np.average(train_loss)
    valid_losses = np.average(valid_loss)
    avg_train_loss.append(train_losses)
    avg_valid_loss.append(valid_losses)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1, train_losses, valid_losses))
    # save model if validation loss has decreased
    train_loss = []
    valid_loss = []
    early_stopping(valid_losses, model)
    if early_stopping.early_stop:
        print('Early Stopping')
        break   
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'fcn_hardnet.pt')
        valid_loss_min = valid_loss

