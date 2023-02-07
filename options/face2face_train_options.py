# -*- coding: utf-8 -*-
"""
This script contains the options used doe the face2face model

@author: Amin Fadaeinejad
"""
import argparse
from .base_options import BaseOptions
from options.train_options import TrainOptions


class Face2FaceTrainOption(BaseOptions):
    """
    This class defines options used for the face2face model
    """
    def initialize(self, parser):
        """
        The first three are for the face2face model itself
        """
        # parser = argparse.ArgumentParser(description='Gather the parameters for the project')
        parser.add_argument('--used_device', type=str, default='none',
                            help='Which device are you using at the moment-> '
                                 '[ YorkU | Ubisoft | laptop ]')
        parser.add_argument('--domain_A', type=str, default='', help="what is the first folder name")
        parser.add_argument('--domain_B', type=str, default='', help="what is the first folder name")
        """
        The default base options for the pix2pix train model
        """
        parser = TrainOptions.initialize(self, parser)
        self.isTrain = True
        return parser

