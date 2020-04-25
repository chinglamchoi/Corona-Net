import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
from skimage.segmentation import find_boundaries
#from skimage.metrics import adapted_rand_error
from sklearn.metrics import adjusted_rand_score
#from loss import DiceLoss
from skimage import io
import unet_6
import unet
import torch.nn as nn

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


class Covid(data.Dataset):
    def __init__(self, imgs, masks, transform=transforms.ToTensor(), img_transform=transforms.Normalize([-377.6907247987177], [574.93976583918584])):
        self.imgs, self.masks = imgs, masks
        self.transform, self.img_transform = transform, img_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img1, mask1 = self.transform(self.imgs[index]).float(), self.transform(self.masks[index]).float()
        img1 = self.img_transform(img1)
        return (img1, mask1)


def DiceLoss(a,b):
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))

def RandLoss(a,b):
    a = (a >= 0.5).float()
    a = a.cpu().numpy().flatten()
    b = b.cpu().numpy().flatten()
    c = adjusted_rand_score(a,b)
    c = (c+1)/2
    return 1 - c



if __name__ == "__main__":
    test_imgs, test_masks = np.load("test_imgs_1.npy"), np.load("test_masks_1.npy")
    testset = Covid(imgs=test_imgs, masks=test_masks)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=12)
    
    a = "cuda:6"
    device = torch.device(a if torch.cuda.is_available() else "cpu")

    net = unet.run_cnn()
    aug = int(input("1 or 0? "))
    ver = int(input("version 1/6: "))
    net = unet.run_cnn() if ver == 1 else unet_6.run_cnn()
    pretrain = input("File path of pretrained model: ")
    if aug > 0:
        checkpoint = torch.load("models_aug/" + pretrain + "/best.pt", map_location="cuda:6") if ver == 1 else torch.load("models_6_aug/" + pretrain + "/best.pt", map_location="cuda:6")
    else:
        checkpoint = torch.load("models/" + pretrain + "/best.pt", map_location="cuda:6") if ver == 1 else torch.load("models_6/" + pretrain + "/best.pt", map_location="cuda:6")
    net.load_state_dict(checkpoint["net"])

    net.to(device)
    net = net.eval()
    #path_ = "./mask_pred/"
    #try:
        #os.mkdir(path_)
    #except:
        #pass

    tot = 0.0
    tot2 = 0.0
    tot_rand = 0.0
    cntcnt = 0
    #a = 0
    with torch.no_grad():
        for img, mask in testloader:
            img, mask = img.to(device), mask.to(device)
            mask_pred = net(img)
            t = torch.sigmoid(mask_pred)
            t_ = (t >= 0.5).float()
            tot += DiceLoss(t_, mask)
            tot_rand += RandLoss(t_, mask)
            t_.to("cpu")
            #torchvision.utils.save_image(t_, path_ + str(cntcnt) + ".jpg")
            cntcnt += 1
    print("Dice Loss:", tot/20) #dice loss
    print("Rand Lcore:", tot_rand/20)
    #print("Rand error: %f | Rand precision: %f | Rand recall: %f "%(tot_rand[0]/759, tot_rand[1]/759, tot_rand[2]/759))
