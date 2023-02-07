# -*- coding: utf-8 -*-
"""
This script contains the dataloader of our model

@author: Amin Fadaeinejad
"""

import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, get_keys
from PIL import Image
import numpy as np
import torch


class Face2FaceDataset(BaseDataset):

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.device = 'cpu' if (opt.gpu_ids == -1) else 'cuda'
        self.dir_A = os.path.join(opt.dataroot, opt.phase, opt.domain_A)  # create a path '/path/to/data/train/albedo'
        self.dir_B = os.path.join(opt.dataroot, opt.phase, opt.domain_B)  # create a path '/path/to/data/train/concat'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        B_path = self.B_paths[index % self.B_size]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # generate random parameters
        transform_params = get_params(self.opt, A_img.size)
        # apply the same transform to both A and B
        transform_A = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        transform_B = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A_keypoints = torch.tensor(np.load(get_keys(A_path)), device='cpu')
        B_keypoints = torch.tensor(np.load(get_keys(B_path)), device='cpu')
        A = transform_A(A_img)
        B = transform_B(B_img)

        return {'A': A, 'B': B, 'A_key': A_keypoints, 'B_key': B_keypoints, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
