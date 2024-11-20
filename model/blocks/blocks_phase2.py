import torch
import torch.nn as nn

from .block_utils.res_block import ResBlock
from .block_utils.cross_attention import Chuncked_Cross_Attn_FM
from .block_utils.cbam import CBAM


class CoherenceAwareAggregation(nn.Module):
    # output: 4 * dim
    def __init__(self, dim):
        super(CoherenceAwareAggregation, self).__init__()

        self.x_branch = nn.Sequential(
            nn.Conv2d(6, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim // 2, use_dropout=False)
        )
        self.y_branch = nn.Sequential(
            nn.Conv2d(6, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim // 2, use_dropout=False)
        )

        self.semantic_branch_x = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim // 2, use_dropout=False)
        )

        self.semantic_branch_y = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim // 2, use_dropout=False)
        )

        self.fuse_x = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(gate_channels=dim, reduction_ratio=dim // 2, pool_types=('avg', 'max'), no_spatial=False),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fuse_y = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(gate_channels=dim, reduction_ratio=dim // 2, pool_types=('avg', 'max'), no_spatial=False),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x_, y_, S0_temp):
        f_x = self.x_branch(x_)
        f_y = self.y_branch(y_)
        fs_x = self.semantic_branch_x(S0_temp)
        fs_y = self.semantic_branch_y(S0_temp)
        f_semantic_x = self.fuse_x(torch.cat((f_x, fs_x), dim=1))
        f_semantic_y = self.fuse_y(torch.cat((f_y, fs_y), dim=1))
        return f_semantic_x, f_semantic_y


class CoherenceInjection(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(CoherenceInjection, self).__init__()

        self.proj_conv = nn.Sequential(
            ResBlock(input_nc * 2, use_dropout=False),
            nn.Conv2d(input_nc * 2, output_nc, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(output_nc),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.cross_attention = Chuncked_Cross_Attn_FM(in_channel=output_nc, r=8, subsample=True, grid=(8, 8))

    def forward(self, f_in, f_semantic):
        f_in = self.proj_conv(torch.cat((f_in, f_semantic), dim=1))
        f_in = self.cross_attention(f_in, f_semantic)
        return f_in
