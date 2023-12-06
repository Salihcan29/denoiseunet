import torch
from torch import nn
import torch.nn.functional as F


class DenoiseUNet(nn.Module):
    def __init__(self):
        super(DenoiseUNet, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)

        # Decoder
        self.dec_conv1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(384, 128, kernel_size=3, padding=1)  # 256 + 128
        self.dec_conv3 = nn.Conv2d(192, 64, kernel_size=3, padding=1)  # 128 + 64
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final conv (without any ReLU)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = F.relu(self.enc_conv1(x))
        enc2 = F.relu(self.enc_conv2(self.max_pool(enc1)))
        enc3 = F.relu(self.enc_conv3(self.max_pool(enc2)))
        enc4 = F.relu(self.enc_conv4(self.max_pool(enc3)))

        # Decoder with skip connections
        dec3 = F.relu(self.dec_conv1(torch.cat([self.up_sample(enc4), enc3], 1)))
        dec2 = F.relu(self.dec_conv2(torch.cat([self.up_sample(dec3), enc2], 1)))
        dec1 = F.relu(self.dec_conv3(torch.cat([self.up_sample(dec2), enc1], 1)))

        # Final conv
        out = self.final_conv(dec1)

        return out

