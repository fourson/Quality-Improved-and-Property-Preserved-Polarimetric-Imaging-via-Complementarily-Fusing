import torch

from base.base_model import BaseModel

from .model_phase1 import DefaultModel as Phase1
from .model_phase2 import DefaultModel as Phase2
from .model_phase3 import DefaultModel as Phase3


class DefaultModel(BaseModel):
    def __init__(self, dim_phase1=16, dim_phase2=32, dim_phase3=32, freeze12=False):
        super(DefaultModel, self).__init__()

        self.Phase1 = Phase1(dim_phase1)
        self.Phase2 = Phase2(dim_phase2)
        self.Phase3 = Phase3(dim_phase3)

        if freeze12:
            # freeze phase1 and phase2 initially
            for p in self.Phase1.parameters():
                p.requires_grad = False
            for p in self.Phase2.parameters():
                p.requires_grad = False

    def forward(self, S0_B, S0_L, S1_L, S2_L, x_B, y_B):
        S0_temp = self.Phase1(S0_B, S0_L, S1_L, S2_L)

        x_temp = S1_L / (S0_temp + 1e-7)
        y_temp = S2_L / (S0_temp + 1e-7)

        x_out, y_out = self.Phase2(x_temp, x_B, y_temp, y_B, S0_temp)
        S1_out = x_out * S0_temp
        S2_out = y_out * S0_temp

        I1_temp = torch.clamp((S0_temp - S1_out) / 2, min=0, max=1)
        I2_temp = torch.clamp((S0_temp - S2_out) / 2, min=0, max=1)
        I3_temp = torch.clamp((S0_temp + S1_out) / 2, min=0, max=1)
        I4_temp = torch.clamp((S0_temp + S2_out) / 2, min=0, max=1)

        I1_out, I2_out, I3_out, I4_out = self.Phase3(I1_temp, I2_temp, I3_temp, I4_temp)

        return S0_temp, x_out, y_out, I1_out, I2_out, I3_out, I4_out
