from torch import nn
import torch
from torch.nn import functional as F
from torchsummary import summary
from timm.models.layers import DropPath, trunc_normal_
from utils.MTSEM import MT_SEM
from utils.MSCAM import MS_CAM
from utils.ConvNeXtV2 import LayerNorm, GRN


class Block(nn.Module):
    def __init__(self, inc, ouc, dp_rate, norm=True):
        super().__init__()
        # Convolution Module
        self.conv = nn.Conv2d(inc, ouc, kernel_size=3, padding=1)
        self.norm_layer = nn.InstanceNorm2d(ouc) if norm else None
        self.activation = nn.GELU()

        # ConvNeXtBlock
        self.dwconv = nn.Conv2d(ouc, ouc, kernel_size=7, padding=3, groups=ouc)  # Depthwise convolution
        self.norm = LayerNorm(ouc, eps=1e-6)
        self.pwconv1 = nn.Linear(ouc, 4 * ouc)  # Pointwise/1x1 convolutions implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * ouc)
        self.pwconv2 = nn.Linear(4 * ouc, ouc)
        self.drop_path = DropPath(dp_rate) if dp_rate > 0. else nn.Identity()

    def forward(self, x):
        # Forward pass through Convolution Module
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.activation(x)

        # Save input for skip connection
        input = x

        # Forward pass through ConvNeXtBlock
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # Add skip connection
        x = input + self.drop_path(x)
        return x


class AquaNeXt(nn.Module):
    def __init__(
            self,
            input_channels: int = 3,
            final_channel: int = 32,
            num_classes=2,
            use_ms_cam=True  # Parameter to enable/disable MS-CAM
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_ms_cam = use_ms_cam  # Initialize use_ms_cam

        dp_rates = [0, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2]

        self.conv1 = Block(input_channels, 32, dp_rate=dp_rates[0])
        self.conv2 = Block(32, 64, dp_rate=dp_rates[1])
        self.conv3 = Block(64, 128, dp_rate=dp_rates[2])
        self.conv4 = Block(128, 256, dp_rate=dp_rates[3])
        self.conv5 = Block(256, 512, dp_rate=dp_rates[4])

        self.conv6 = Block(768, 256, dp_rate=dp_rates[5])
        self.conv7 = Block(384, 128, dp_rate=dp_rates[6])
        self.conv8 = Block(192, 64, dp_rate=dp_rates[7])
        self.conv9 = Block(96, 32, dp_rate=dp_rates[8])

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=4)
        self.MT_SEM = MT_SEM()

        if use_ms_cam:  # Initialize MS-CAM modules if use_ms_cam is True
            self.ms_cam4 = MS_CAM(channels=256)
            self.ms_cam3 = MS_CAM(channels=128)
            self.ms_cam2 = MS_CAM(channels=64)
            self.ms_cam1 = MS_CAM(channels=32)
        else:  # Use identity layers if MS-CAM is disabled
            self.ms_cam4 = nn.Identity()
            self.ms_cam3 = nn.Identity()
            self.ms_cam2 = nn.Identity()
            self.ms_cam1 = nn.Identity()

        self.conv_final_mask = nn.Conv2d(final_channel, num_classes, 1)
        self.conv_final_edge = nn.Conv2d(final_channel, num_classes, 1)
        self.conv_final_dist = nn.Conv2d(final_channel, 1, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        x2 = self.pool1(x2)

        x3 = self.conv3(x2)
        x3 = self.pool1(x3)

        x4 = self.conv4(x3)
        x4 = self.pool1(x4)

        x5 = self.conv5(x4)
        x5 = self.pool2(x5)

        x_6 = self.upsample2(x5)

        x4 = self.ms_cam4(x4)
        x6 = self.conv6(torch.cat([x_6, x4], 1))
        x6 = self.upsample1(x6)

        x3 = self.ms_cam3(x3)
        x7 = self.conv7(torch.cat([x6, x3], 1))
        x7 = self.upsample1(x7)

        x2 = self.ms_cam2(x2)
        x8 = self.conv8(torch.cat([x7, x2], 1))
        x8 = self.upsample1(x8)

        x1 = self.ms_cam1(x1)
        x9 = self.conv9(torch.cat([x8, x1], 1))
        x_out_tasks = self.MT_SEM(x9)

        x_out1 = self.conv_final_mask(x_out_tasks[0])
        x_out2 = self.conv_final_edge(x_out_tasks[1])
        x_out3 = self.conv_final_dist(x_out_tasks[2])

        mask = F.log_softmax(x_out1, dim=1)
        contour = F.log_softmax(x_out2, dim=1)
        dist = x_out3
        return [mask, contour, dist]


if __name__ == '__main__':
    model = AquaNeXt(num_classes=2, use_ms_cam=False)
    batch_size = 2
    channels = 3
    height = 384
    width = 384
    input_tensor = torch.randn(batch_size, channels, height, width)

    # Run model inference
    with torch.no_grad():
        output = model(input_tensor)
    model.cuda()
    # Print model summary using torchsummary
    summary(model, input_size=(3, 512, 512))
