from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        #self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dp1 = nn.Dropout(inplace=False)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        #self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dp2 = nn.Dropout(inplace=False)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        #self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dp3 = nn.Dropout(inplace=False)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        #self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dp4 = nn.Dropout(inplace=False)
        self.encoder5 = UNet._block(features*8, features*16, name="enc5")

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dp5 = nn.Dropout(inplace=False)
        self.encoder6 = UNet._block(features*16, features*32, name="enc6")

        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dp6 = nn.Dropout(inplace=False)        
        
        self.bottleneck = UNet._block(features*32, features*64, name="bottleneck")
        #self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv6 = nn.ConvTranspose2d(features*64, features*32, kernel_size=2, stride=2)
        self.decoder6 = UNet._block((features*32)*2, features*32, name="dec6")
        self.dp66 = nn.Dropout(inplace=False)

        self.upconv5 = nn.ConvTranspose2d(features*32, features*16, kernel_size=2, stride=2)
        self.decoder5 = UNet._block((features*16)*2, features*16, name="dec5")
        self.dp55 = nn.Dropout(inplace=False)
        
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        #self.bn44 = nn.BatchNorm2d(256)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.dp44 = nn.Dropout(inplace=False)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        #self.bn33 = nn.BatchNorm2d(128)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.dp33 = nn.Dropout(inplace=False)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        #self.bn22 = nn.BatchNorm2d(64)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.dp22 = nn.Dropout(inplace=False)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        #self.bn11 = nn.BatchNorm2d(32)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        ##self.dp11 = nn.Dropout(inplace=False)
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.dp1(self.pool1(enc1)))
        enc3 = self.encoder3(self.dp2(self.pool2(enc2)))
        enc4 = self.encoder4(self.dp3(self.pool3(enc3)))
        enc5 = self.encoder5(self.dp4(self.pool4(enc4)))
        enc6 = self.encoder6(self.dp5(self.pool5(enc5)))
        bottleneck = self.bottleneck(self.dp6(self.pool6(enc6)))

        dec6 = self.upconv6(bottleneck)
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec6 = self.dp66(self.decoder6(dec6))

        dec5 = self.upconv5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.dp55(self.decoder5(dec5))

        dec4 = self.upconv4(dec5)
        #dec4 = self.bn44(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dp44(self.decoder4(dec4))
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dp33(self.decoder3(dec3))
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dp22(self.decoder2(dec2))
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)
        return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.LeakyReLU(inplace=True)),
                ]
            )
        )
def run_cnn():
    return UNet()
"""
a = torch.ones(1, 1, 512, 512)
m = run_cnn()
m(a)
"""
