import torch

from .loss_utils.l12 import l1, l2

tag = 'loss_phase3'


def l1_and_pr(I1_out, I1, I2_out, I2, I3_out, I3, I4_out, I4, **kwargs):
    I_out_cat = torch.cat((I1_out, I2_out, I3_out, I4_out), dim=1)
    I_cat = torch.cat((I1, I2, I3, I4), dim=1)

    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = l1(I_out_cat, I_cat) * l1_loss_lambda
    print(f'in {tag}, l1_loss: {l1_loss.item()}')

    pr_loss_lambda = kwargs.get('pr_loss_lambda', 1)
    pr_loss = l2(I1_out + I3_out, I2_out + I4_out) * pr_loss_lambda
    print(f'in {tag}, pr_loss: {pr_loss.item()}')

    return l1_loss + pr_loss
