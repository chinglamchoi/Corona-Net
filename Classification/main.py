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
        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        #self.model = EfficientNet.from_name("efficientnet-b0")
        #for param in self.model.parameters():
            #param.requires_grad = True
        self.dp1 = nn.Dropout(p=0.4)
        #self.ln1 = nn.Linear(in_features=1000, out_features=500, bias=True)
        self.ln1 = nn.Linear(in_features=1000, out_features=1, bias=True)
        #self.dp2 = nn.Dropout(p=0.1)
        #self.ln2 = nn.Linear(in_features=128, out_features=10, bias=True)
        #self.dp3 = nn.Dropout(p=0.1)
        #self.ln3 = nn.Linear(in_features=10, out_features=1, bias=True)
    def forward(self, x):
        out = F.relu(self.model(x))
        out = out.view(out.size(0), -1)
        out = self.ln1(self.dp1(F.relu(out)))
        #out = self.ln2(self.dp2(F.relu(out)))
        #out = self.ln3(self.dp3(F.relu(out)))
        return out

def run_cnn():
    return CoronaNet()

class Covid(data.Dataset):
    def __init__(self, imgs, lbs, transform=transforms.ToTensor(), img_transform=transforms.Normalize([-634.8493269908545], [548.0778525978939])):
        self.imgs, self.lbs = imgs, lbs
        self.transform, self.img_transform = transform, img_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img = self.transform(self.imgs[index]).float()
        img = self.img_transform(img)
        lb = torch.tensor(self.lbs[index]).float()
        img = img.repeat(3,1,1)
        return (img, lb)

def lr_change(alpha):
    global optimizer
    alpha /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha
    print("Lr changed to " + str(alpha))

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
    trainset = Covid(imgs=train_imgs, lbs=train_lbs)
    testset = Covid(imgs=test_imgs, lbs=test_lbs)

    class_counts = [864.0, 555.0] ##
    weights_per_class = [1419.0/class_counts[i] for i in range(len(class_counts))] ##
    weights1 = [weights_per_class[train_lbs[i]] for i in range(1419)] ##
    sampler = data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights1), 1419) ##
    trainloader = data.DataLoader(trainset, batch_size=11, shuffle=False, sampler=sampler, num_workers=12) ##
    testloader = data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=12) ##

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
        epoch_loss = 0.0
        for img, lb in trainloader:
            lb = lb.reshape(lb.size(0), 1)
            img, lb = img.to(device), lb.to(device)
            lb_hat = net(img)
            loss = criterion1(lb_hat, lb)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(epoch_loss/129) ##
        print("Epoch" + str(epoch) + " Train BCE Loss:", epoch_loss/129) ##

        net = net.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for img, lb in testloader:
                lb = lb.reshape(lb.size(0), 1)
                img, lb = img.to(device), lb.to(device)
                lb_hat = net(img)
                loss = criterion1(lb_hat, lb)
                epoch_loss += loss.item()
        epoch_loss /= 15.7
        test_loss.append(epoch_loss)
        print("Epoch" + str(epoch) + " Test BCE Loss:", epoch_loss) ##

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
            path_ = "./models_class/" + args.opt + "_lr" + str(args.lr) + "_stp" + str(args.stp) + "/" 
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
