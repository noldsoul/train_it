import os
import time
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns
from models.fcn_hardnet import hardnet
import torchvision
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def give_color_to_seg_img(seg,n_classes):
    '''
    seg : (input_width,input_height,3)
    '''
    seg_img = np.zeros( (seg.shape[0],seg.shape[1],3) ).astype('float')
    colors = sns.color_palette("hls", n_classes)
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0] ))
        seg_img[:,:,1] += (segc*( colors[c][1] ))
        seg_img[:,:,2] += (segc*( colors[c][2] ))
    return(seg_img)


device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = hardnet(n_classes = 5)
# model = torchvision.models.segmentation.fcn_resnet50(pretrained = False, num_classes = 5)
model.load_state_dict(torch.load('/home/chandradeep_p/??????????????????????????/saved_models/fchardnet_50_BDD_IOU.pt', map_location = torch.device('cpu')))
# model.load_state_dict(torch.load('/home/deeplearning/Desktop/chandradeep/fchardnet_200_flux.pt'))
model.eval()
print(device)
model.to(device)
video_path = '/home/deeplearning/Desktop/chandradeep/Segmentation/test_videos/Video 3.avi'
cap =cv2.VideoCapture(video_path)
while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (224, 224))
        cv2.imshow('frame', frame)
        t0 = time.time()
        frame = torch.from_numpy(frame).float().permute([2,0,1])
        frame = frame.unsqueeze(0)
        frame = frame.cuda()
        print(type(frame))
        output = model(frame)
        output = output.cpu()
        output = output[0].argmax(0).numpy()
        print(output.shape)
        output = give_color_to_seg_img(output, 5)
        # plt.imshow(output)
        # plt.show()
        cv2.imshow('output', output)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
                break
        print("FPS = ", 1/(time.time()-t0))
        print("##############End################")