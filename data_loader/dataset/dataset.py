import os
import fnmatch

import cv2
from torch.utils.data import Dataset


class WithGroundTruthDataset(Dataset):
    """
        as input:
        L1, L2, L3, L4: four polarized images with short-exposure (noisy, pre-amplified), [0, 1], as float32
        L1, L2, L3, L4: four polarized images with long-exposure (blurry), [0, 1], as float32

        as target:
        I1, I2, I3, I4: four enhanced polarized images, [0, 1], as float32
    """

    def __init__(self, data_dir, transform=None):
        self.L1_dir = os.path.join(data_dir, 'L1')
        self.L2_dir = os.path.join(data_dir, 'L2')
        self.L3_dir = os.path.join(data_dir, 'L3')
        self.L4_dir = os.path.join(data_dir, 'L4')

        self.B1_dir = os.path.join(data_dir, 'B1')
        self.B2_dir = os.path.join(data_dir, 'B2')
        self.B3_dir = os.path.join(data_dir, 'B3')
        self.B4_dir = os.path.join(data_dir, 'B4')

        self.I1_dir = os.path.join(data_dir, 'I1')
        self.I2_dir = os.path.join(data_dir, 'I2')
        self.I3_dir = os.path.join(data_dir, 'I3')
        self.I4_dir = os.path.join(data_dir, 'I4')

        self.names = [file_name for file_name in fnmatch.filter(os.listdir(self.L1_dir), '*.png')]

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        # as input:
        # (H, W, 3)
        L1 = cv2.cvtColor(cv2.imread(os.path.join(self.L1_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L2 = cv2.cvtColor(cv2.imread(os.path.join(self.L2_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L3 = cv2.cvtColor(cv2.imread(os.path.join(self.L3_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L4 = cv2.cvtColor(cv2.imread(os.path.join(self.L4_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8

        # (H, W, 3)
        B1 = cv2.cvtColor(cv2.imread(os.path.join(self.B1_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        B2 = cv2.cvtColor(cv2.imread(os.path.join(self.B2_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        B3 = cv2.cvtColor(cv2.imread(os.path.join(self.B3_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        B4 = cv2.cvtColor(cv2.imread(os.path.join(self.B4_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8

        # as target:
        # (H, W, 3)
        I1 = cv2.cvtColor(cv2.imread(os.path.join(self.I1_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        I2 = cv2.cvtColor(cv2.imread(os.path.join(self.I2_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        I3 = cv2.cvtColor(cv2.imread(os.path.join(self.I3_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        I4 = cv2.cvtColor(cv2.imread(os.path.join(self.I4_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8

        if self.transform:
            L1 = self.transform(L1)
            L2 = self.transform(L2)
            L3 = self.transform(L3)
            L4 = self.transform(L4)

            B1 = self.transform(B1)
            B2 = self.transform(B2)
            B3 = self.transform(B3)
            B4 = self.transform(B4)

            I1 = self.transform(I1)
            I2 = self.transform(I2)
            I3 = self.transform(I3)
            I4 = self.transform(I4)

        return {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4,
                'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4,
                'I1': I1, 'I2': I2, 'I3': I3, 'I4': I4,
                'name': name}


class WithoutGroundTruthDataset(Dataset):
    """
        as input:
        L1, L2, L3, L4: four polarized images with short-exposure (noisy, pre-amplified), [0, 1], as float32
        L1, L2, L3, L4: four polarized images with long-exposure (blurry), [0, 1], as float32
    """

    def __init__(self, data_dir, transform=None):
        self.L1_dir = os.path.join(data_dir, 'L1')
        self.L2_dir = os.path.join(data_dir, 'L2')
        self.L3_dir = os.path.join(data_dir, 'L3')
        self.L4_dir = os.path.join(data_dir, 'L4')

        self.B1_dir = os.path.join(data_dir, 'B1')
        self.B2_dir = os.path.join(data_dir, 'B2')
        self.B3_dir = os.path.join(data_dir, 'B3')
        self.B4_dir = os.path.join(data_dir, 'B4')

        self.names = [file_name for file_name in fnmatch.filter(os.listdir(self.L1_dir), '*.png')]

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        # as input:
        # (H, W, 3)
        L1 = cv2.cvtColor(cv2.imread(os.path.join(self.L1_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L2 = cv2.cvtColor(cv2.imread(os.path.join(self.L2_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L3 = cv2.cvtColor(cv2.imread(os.path.join(self.L3_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        L4 = cv2.cvtColor(cv2.imread(os.path.join(self.L4_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8

        # (H, W, 3)
        B1 = cv2.cvtColor(cv2.imread(os.path.join(self.B1_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        B2 = cv2.cvtColor(cv2.imread(os.path.join(self.B2_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        B3 = cv2.cvtColor(cv2.imread(os.path.join(self.B3_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8
        B4 = cv2.cvtColor(cv2.imread(os.path.join(self.B4_dir, name)), cv2.COLOR_BGR2RGB)  # [0, 255] uint8

        if self.transform:
            L1 = self.transform(L1)
            L2 = self.transform(L2)
            L3 = self.transform(L3)
            L4 = self.transform(L4)

            B1 = self.transform(B1)
            B2 = self.transform(B2)
            B3 = self.transform(B3)
            B4 = self.transform(B4)

        return {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4,
                'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4,
                'name': name}
