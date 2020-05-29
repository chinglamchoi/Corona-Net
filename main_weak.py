import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import os
from efficientnet import EfficientNet
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

class CoronaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b1")
        self.dp1 = nn.Dropout(p=0.4)
        self.conv = nn.ConvTranspose2d(1280, 1, 497)
        self.dp2 = nn.Dropout(p=0.4)
        self.p1 = nn.AdaptiveAvgPool2d(1)
        self.ln1 = nn.Linear(1, 1, bias=True)
    def forward(self, x):
        out = F.relu(self.model.extract_features(x))
        out = self.dp2(F.relu(self.conv(self.dp1(out))))
        out1 = self.ln1(self.p1(out))
        return out, out1

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

def run_cnn():
    return CoronaNet()

def DiceLoss(a,b):
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return 1- ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))

class Covid(data.Dataset):
    def __init__(self, imgs, lbs, msks, transform=transforms.ToTensor(), img_transform=transforms.Normalize([-634.8493269908545], [548.0778525978939])):
        self.imgs, self.lbs, self.msks = imgs, lbs, msks
        self.transform, self.img_transform = transform, img_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = self.transform(self.imgs[index]).float()
        img = self.img_transform(img)
        lb = torch.tensor(self.lbs[index]).float()
        msks = torch.tensor(self.msks[index]).float()
        img = img.repeat(3,1,1)
        return (img, lb, msks)

def lr_change(alpha):
    global optimizer
    alpha /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha
    print("Lr changed to " + str(alpha))
    return alpha

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pre", metavar="PRE", type=str, default=None, dest="pre")
    parser.add_argument("-lr", metavar="LR", type=float, default=0.001, dest="lr")
    parser.add_argument("-eps", metavar="E", type=int, default=400, dest="eps")
    parser.add_argument("-wd", metavar="WD", type=float, default=5e-4, dest="wd")
    parser.add_argument("-m", metavar="M", type=float, default=0, dest="m")
    parser.add_argument("-opt", metavar="OPT", type=str, default="Adam", dest="opt")
    parser.add_argument("-cuda", metavar="CUDA", type=int, default=0, dest="cuda")
    parser.add_argument("-stp", metavar="STP", type=int, default=100, dest="stp")
    args = parser.parse_args()

    train_imgs, train_lbs, test_imgs, test_lbs = np.load("data/COVID_train_imgs.npy"), np.load("data/COVID_train_lbs.npy"), np.load("data/COVID_test_imgs.npy"), np.load("data/COVID_test_lbs.npy") ##
    train_masks, test_masks = np.load("data/COVID_train_msks.npy"), np.load("data/COVID_test_msks.npy")
    trainset = Covid(imgs=train_imgs, lbs=train_lbs, msks=train_masks)
    testset = Covid(imgs=test_imgs, lbs=test_lbs, msks=test_masks)

    class_counts = [864.0, 555.0] ##
    weights_per_class = [1419.0/class_counts[i] for i in range(len(class_counts))] ##
    weights1 = [weights_per_class[train_lbs[i]] for i in range(1419)] ##
    sampler = data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights1), 1419) ##
    trainloader = data.DataLoader(trainset, batch_size=5, shuffle=False, sampler=sampler, num_workers=12) ##
    testloader = data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=12) ##

    a = "cuda:" + str(args.cuda)
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    criterion1 = nn.BCEWithLogitsLoss().to(device)
    net = run_cnn()
    val = False
    alpha = args.lr
    if args.pre is not None:
        checkpoint = torch.load(args.pre)
        net.load_state_dict(checkpoint["net"])
        alpha = checkpoint["alpha"]
    net.to(device)
    net._modules.get("conv").register_forward_hook(hook_feature)

    if args.opt == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    else:
        optimizer = optimizer = optim.RMSprop(net.parameters(), lr=alpha, weight_decay=args.wd)
    if val:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    train_loss, test_loss = [], []
    start_ = checkpoint["epoch"] if val else 1
    best_loss = checkpoint["loss"] if val else 100
    epochs = checkpoint["epoch"]+args.eps if val else args.eps

    try:
        os.mkdir("models")
    except:
        pass

    for epoch in range(start_, epochs+1):
        if epoch % args.stp == 0 and epoch != epochs:
            alpha = lr_change(alpha)
        net = net.train()
        epoch_loss, epoch_loss1 = 0.0, 0.0
        for img, lb, msk in trainloader:
            weights = np.squeeze(list(net.parameters())[-2].cpu().data.numpy())
            feature_blobs = []
            lb = lb.reshape(lb.size(0), 1, 1, 1)
            img, msk, lb = img.to(device), msk.to(device), lb.to(device)
            mask_pred, lb_hat = net(img)
            loss = criterion1(lb_hat, lb)
            CAMs = returnCAM(feature_blobs[0], weights)
            t = torch.sigmoid(CAMs).to(device)
            loss1 = DiceLoss(t, msk).item()
            epoch_loss1 += loss1
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss, epoch_loss1 = epoch_loss / 283.8, epoch_loss1 / 283.8
        train_loss.append(epoch_loss) ##
        print("Epoch" + str(epoch) + " Train BCE Loss:", epoch_loss)
        print("Epoch" + str(epoch) + " Train Dice Loss", epoch_loss1)

        net = net.eval()
        epoch_loss, epoch_loss1 = 0.0, 0.0
        with torch.no_grad():
            for img, lb, msk in testloader:
                weights = np.squeeze(list(net.parameters())[-2].cpu().data.numpy())
                feature_blobs = []
                lb = lb.reshape(lb.size(0), 1, 1, 1)
                img, msk, lb = img.to(device), msk.to(device), lb.to(device)
                mask_pred, lb_hat = net(img)
                loss = criterion1(lb_hat, lb)
                CAMs = returnCAM(feature_blobs[0], weights).to(device)
                t = torch.sigmoid(CAMs).to(device)
                loss1 = DiceLoss(t, msk).item()
                epoch_loss1 += loss1
                epoch_loss += loss.item()
        del t
        del loss
        del CAMs
        del loss1
        epoch_loss, epoch_loss1 = epoch_loss/31.4, epoch_loss1/31.4
        test_loss.append(epoch_loss)
        print("Epoch" + str(epoch) + " Test BCE Loss:", epoch_loss)
        print("Epoch" + str(epoch) + " Test Dice Loss:", epoch_loss1)

        if epoch_loss < best_loss:
            valid = True
            best_loss = epoch_loss
            print("New best test loss!")
        else:
            valid = False
        print("\n")

        if valid:
            for param_group in optimizer.param_groups:
                alpha_ = param_group["lr"]
            state = {
                "net":net.state_dict(),
                "loss": epoch_loss,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "alpha": alpha_
            }
            path_ = "./models_class1/" + args.opt + "_lr" + str(args.lr) + "_stp" + str(args.stp) + "/" 
            try:
                os.mkdir(path_)
            except:
                pass
            torch.save(state, str(path_ + "best.pt"))
        
        if epoch == epochs:
            fig = plt.figure()
            plt.plot(train_loss, label="Train")
            plt.plot(test_loss, label="Test")
            plt.xlabel("Epochs")
            plt.ylabel("BCE Loss")
            plt.title("Train-Test BCE Loss")
            fig.savefig(path_ + "bce.png")
