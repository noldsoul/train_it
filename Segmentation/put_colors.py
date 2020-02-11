import os
import torch
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 

mask = Image.open('/home/chandradeep_p/Desktop/datasets/Flux_data/dataset2.0/masks/frame10677.png')
mask = np.array(mask)
n_classes = 47
def put_palette(n_classes, mask ):
        # create a color pallette, selecting a color for each class
        palette = torch.tensor([4 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(n_classes)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")
        # plot the semantic segmentation predictions of 21 classes in each color
        # r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        mask = Image.fromarray(mask)
        mask.putpalette(colors)
        return mask

# put_palette(47, mask).show()