# sometimes training with Dice Loss directly yields better results

import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
#import unet_4
import unet_6
import unet
from sklearn.metrics import adjusted_rand_score

from skimage import io
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import random


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

def lr_change(alpha):
    global optimizer
    alpha /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha
    print("Lr changed to " + str(alpha))
    return alpha

def DiceLoss(a,b):
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return 1- ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))

def RandLoss(a,b):
    a = (a>=0.5).float()
    a = a.cpu().numpy().flatten()
    b = b.cpu().numpy().flatten()
    c = adjusted_rand_score(a,b)
    c = (c+1)/2
    return 1-c

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        self.outc1 = nn.AdaptiveAvgPool2d((1,1))
        self.outc2 = nn.Linear(1,1)
    def forward(self, x):
        out = self.outc1(x)
        out = self.outc2(out.view(out.size(0), -1))
        return x, out

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
    def forward(self, inputt, target):
        logit = torch.sigmoid(inputt)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = -1 * target * torch.log(logit)
        loss = loss * (1 - logit) ** self.gamma # focal loss
        return loss.sum()

def hook_feature(module, input, output):
    feature_blobs.append(output.cpu().data.numpy())

def returnCAM(feature_conv, weight):
    output_cam = []
    for i in range(len(feature_conv)):
        _ = weight.dot(feature_conv[i, :, :, :].reshape(1, 512*512))
        _ = _.reshape(1, 512, 512)
        _ = torch.from_numpy(_)
        output_cam.append(_)
    return torch.stack(output_cam)

def weight_init(m):
    if type(m) == nn.Linear:
        #nn.init.xavier_uniform(m.weight)
        m.weight.data.uniform_(0.0, 0.0)
        m.bias.data.fill_(0)

def run_cnn2():
    return UNet2()

if __name__ == "__main__":
    #mean, std = [-402.0083926882452], [481.1704205180789] 601
    #mean, std = [-377.6907247987177], [474.93976583918584]

    parser = argparse.ArgumentParser()
    parser.add_argument("-pre", metavar="PRE", type=str, default=None, dest="pre")
    parser.add_argument("-lr", metavar="LR", type=float, default=0.001, dest="lr")
    parser.add_argument("-eps", metavar="E", type=int, default=400, dest="eps")
    parser.add_argument("-wd", metavar="WD", type=float, default=5e-4, dest="wd")
    parser.add_argument("-m", metavar="M", type=float, default=0, dest="m")
    parser.add_argument("-opt", metavar="OPT", type=str, default="Adam", dest="opt")
    parser.add_argument("-cuda", metavar="CUDA", type=int, default=0, dest="cuda")
    parser.add_argument("-mul", metavar="MUL", type=bool, default=False, dest="mul") #same thing as GAP
    parser.add_argument("-rec", metavar="REC", type=bool, default=False, dest="rec")
    parser.add_argument("-stp", metavar="STP", type=int, default=100, dest="stp")
    parser.add_argument("-sam", metavar="SAM", type=bool, default=False, dest="sam")
    parser.add_argument("-ver", metavar="V", type=int, default=1, dest="ver")
    #{1: default 5 encoder unet.py, 4: 4 encoders unet_4.py}
    args = parser.parse_args()

    train_imgs, train_masks, test_imgs, test_masks = np.load("train_imgs_1aug.npy"), np.load("train_masks_1binaug.npy"), np.load("test_imgs_1.npy"), np.load("test_masks_1.npy")
    trainset = Covid(imgs=train_imgs, masks=train_masks)
    testset = Covid(imgs=test_imgs, masks=test_masks)

    trainloader = data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=12)
    testloader = data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=12)

    a = "cuda:" + str(args.cuda)
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    criterion1 = nn.BCEWithLogitsLoss().to(device)
    #criterion1 = nn.Sigmoid().to(device)
    #criterion2 = FocalLoss().to(device)
    if args.mul:
        net1 = unet.UNet() if args.ver == 1 else unet_6.UNet()
    else:
        net = unet.run_cnn() if args.ver == 1 else unet_6.run_cnn()
    vall = False
    if args.pre is not None:
        checkpoint = torch.load(args.pre)
        if args.mul:
            net1.load_state_dict(checkpoint["net"])
            for child in net1.children():
                for param in child.parameters():
                    param.requires_grad = True
            net2 = run_cnn2()
            for child in net2.children():
                for param in child.parameters():
                    param.requires_grad = True
            with torch.no_grad():
                net2.outc2.weight = nn.Parameter(torch.zeros(net2.outc2.weight.size()))
            #net2.parameters()[0] = torch.zeros(net2.parameters()[0].size())
            #net2.parameters()[1] = torch.zeros(net2.parameters()[1].size())
            #net2.apply(weight_init)
            net = nn.Sequential(net1, net2)
            #for param in net.parameters():
                #param.requires_grad = True
        else:
            net.load_state_dict(checkpoint["net"])
            vall = True #only for non-GAP pretrained
    else:
        if args.mul:
            net2 = run_cnn2()
            net2.apply(weight_init)
            net = nn.Sequential(net1, net2)
            #raise AssertionError("Pretrained model cannot be null with GAP")
    net.to(device)
    if args.mul:
        net._modules["0"]._modules.get("conv").register_forward_hook(hook_feature)

    best_loss = checkpoint["loss"] if vall else 100

    alpha = checkpoint["alpha"] if vall else args.lr
    if args.opt == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=alpha, weight_decay=args.wd)
        #optimizer = optim.Adam(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=alpha, weight_decay=args.wd, momentum=args.m)
        #optimizer = optim.SGD(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    if vall:
        optimizer.load_state_dict(checkpoint["optimizer"])
    train_loss, val_loss = [], []
    train_loss1 = []
    start_ = checkpoint["epoch"] if vall else 1 
    epochs = checkpoint["epoch"]+args.eps if vall else args.eps
    for epoch in range(start_, epochs+1):
        if epoch % args.stp == 0 and epoch != epochs:
            alpha = lr_change(alpha)
        net = net.train()
        epoch_loss, epoch_loss1 = 0.0, 0.0
        for img, mask in trainloader:
            mask_type = torch.float32
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            if args.mul:
                weights = np.squeeze(list(net.parameters())[-2].cpu().data.numpy())
                feature_blobs = []
                mask_pred, _ = net(img)
                CAMs = returnCAM(feature_blobs[0], weights).to(device)
                t = torch.sigmoid(CAMs)
                loss = criterion1(CAMs, mask)
                #loss.requires_grad = True
                #t = torch.sigmoid(CAMs)
            else:
                mask_pred = net(img)
                #loss = criterion1(mask_pred, mask)
                t = torch.sigmoid(mask_pred)
                loss = criterion1(mask_pred, mask)
                #t = torch.sigmoid(mask_pred)
            #print(loss)
            epoch_loss += loss.item()
            epoch_loss1 += DiceLoss(t, mask).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(epoch_loss/192)
        train_loss1.append(epoch_loss1/192)
        print("Epoch" + str(epoch) + " Train BCE Loss:", epoch_loss/192)
        print("Epoch" + str(epoch) + " Train Dice Loss:", epoch_loss1/192)
        
        net = net.eval()
        tot_val = 0.0
        with torch.no_grad():
            for img, mask in testloader:
                mask_type = torch.float32
                img, mask = (img.to(device), mask.to(device, dtype=mask_type))
                if args.mul:
                    weights = np.squeeze(list(net.parameters())[-2].cpu().data.numpy())
                    feature_blobs = []
                    mask_pred, _ = net(img)
                    CAMs = returnCAM(feature_blobs[0], weights).to(device)
                    t = torch.sigmoid(CAMs)
                else:    
                    mask_pred = net(img)
                    t = torch.sigmoid(mask_pred)
                tot_val += DiceLoss(t, mask).item()
        loss_ = tot_val/4
        print("Epoch" + str(epoch) + " Val Loss:", loss_)
        if loss_ < best_loss:
            valid = True
            print("New best test loss!")
            best_loss = loss_
        else:
            valid = False
        val_loss.append(loss_)
        print("\n")
        
        if valid:
            for param_group in optimizer.param_groups:
                alpha_ = param_group["lr"]
            state = {
                "net":net.state_dict(),
                "loss": loss_,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "alpha": alpha_
            }
            if args.mul:
                path_ = "./models/gap_" + args.opt + "_lr" + str(args.lr) if args.ver == 1 else "./models_6/gap_" + args.opt + "_lr" + str(args.lr)
            else:
                path_ = "./models/aug_vanilla" + args.opt + "_lr" + str(args.lr) if args.ver == 1 else "./models_6/aug_vanilla_" + args.opt + "_lr" + str(args.lr)
            path_ = path_ + "_sam" if args.sam else path_
            path_ += "/"
            try:
                os.mkdir(path_)
            except:
                pass
            torch.save(state, str(path_ + "best.pt"))
        if epoch == epochs:
            fig = plt.figure()
            plt.plot(train_loss1, label="Train")
            plt.plot(val_loss, label="Val")
            plt.xlabel("Epochs")
            plt.ylabel("Dice Loss")
            plt.title("Train-Val Loss")
            fig.savefig(path_+ "train_val.png")
            fig = plt.figure()
            plt.plot(train_loss, label="Train")
            plt.xlabel("Epochs")
            plt.ylabel("BCE Loss")
            plt.title("Train Loss")
            fig.savefig(path_ + "train.png")
            print("Saved plots")
