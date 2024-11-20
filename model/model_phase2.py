import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_model import BaseModel
from .blocks.blocks_phase2 import CoherenceAwareAggregation, CoherenceInjection
from .blocks.block_utils.res_block import ResBlock
from .blocks.block_utils.dense_block import DenseBlock


class DefaultModel(BaseModel):
    def __init__(self, dim=32):
        super(DefaultModel, self).__init__()

        self.x_encoder = nn.Sequential(
            nn.Conv2d(6, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.y_encoder = nn.Sequential(
            nn.Conv2d(6, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.CAG = CoherenceAwareAggregation(dim=dim)

        self.CI1_x = CoherenceInjection(input_nc=dim * 4, output_nc=dim * 4)
        self.CI2_x = CoherenceInjection(input_nc=dim * 4, output_nc=dim * 4)
        self.CI3_x = CoherenceInjection(input_nc=dim * 4, output_nc=dim * 4)

        self.CI1_y = CoherenceInjection(input_nc=dim * 4, output_nc=dim * 4)
        self.CI2_y = CoherenceInjection(input_nc=dim * 4, output_nc=dim * 4)
        self.CI3_y = CoherenceInjection(input_nc=dim * 4, output_nc=dim * 4)

        self.x_decoder = nn.Sequential(
            DenseBlock(in_channel=dim * 4, growth_rate=dim, n_blocks=3),
            nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=4, stride=2, padding=1),
            ResBlock(dim * 2, use_dropout=False),
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=4, stride=2, padding=1),
            ResBlock(dim, use_dropout=False)
        )
        self.y_decoder = nn.Sequential(
            DenseBlock(in_channel=dim * 4, growth_rate=dim, n_blocks=3),
            nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=4, stride=2, padding=1),
            ResBlock(dim * 2, use_dropout=False),
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=4, stride=2, padding=1),
            ResBlock(dim, use_dropout=False)
        )

        self.out_block_x = nn.Conv2d(dim, 3, kernel_size=1, stride=1)
        self.out_block_y = nn.Conv2d(dim, 3, kernel_size=1, stride=1)

    def forward(self, x_temp, x_B, y_temp, y_B, S0_temp):
        # normalize
        x_temp = (x_temp + 1) / 2
        x_B = (x_B + 1) / 2
        y_temp = (y_temp + 1) / 2
        y_B = (y_B + 1) / 2
        S0_temp = S0_temp / 2

        # padding
        x_temp = F.pad(x_temp, pad=[16, 16, 16, 16], mode='reflect')
        x_B = F.pad(x_B, pad=[16, 16, 16, 16], mode='reflect')
        y_temp = F.pad(y_temp, pad=[16, 16, 16, 16], mode='reflect')
        y_B = F.pad(y_B, pad=[16, 16, 16, 16], mode='reflect')
        S0_temp = F.pad(S0_temp, pad=[16, 16, 16, 16], mode='reflect')

        x_ = torch.cat([x_temp, x_B], dim=1)
        y_ = torch.cat([y_temp, y_B], dim=1)

        # multi-scale xy feature
        f_x = self.x_encoder(x_)
        f_y = self.y_encoder(y_)

        # coherence volumes
        coherence_volume_x, coherence_volume_y = self.CAG(x_, y_, S0_temp)

        # coherence injection x
        f_inner1_x = self.CI1_x(f_x, coherence_volume_x)
        f_inner2_x = self.CI2_x(f_inner1_x, coherence_volume_x)
        f_inner3_x = self.CI3_x(f_inner2_x, coherence_volume_x)

        # coherence injection y
        f_inner1_y = self.CI1_y(f_y, coherence_volume_y)
        f_inner2_y = self.CI2_y(f_inner1_y, coherence_volume_y)
        f_inner3_y = self.CI3_y(f_inner2_y, coherence_volume_y)

        # decoding xy features
        output_feature_x = self.x_decoder(f_inner3_x)
        output_feature_y = self.y_decoder(f_inner3_y)

        # out x
        x_out = self.out_block_x(output_feature_x) + x_temp
        x_out = torch.clamp(x_out, min=0, max=1)

        # out y
        y_out = self.out_block_y(output_feature_y) + y_temp
        y_out = torch.clamp(y_out, min=0, max=1)

        # denormalize
        x_out = x_out * 2 - 1
        y_out = y_out * 2 - 1

        # cropping
        x_out = x_out[:, :, 16:-16, 16:-16]
        y_out = y_out[:, :, 16:-16, 16:-16]

        return x_out, y_out
