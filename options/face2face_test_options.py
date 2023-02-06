# -*- coding: utf-8 -*-
"""
This script contains the options used doe the face2face model

@author: Amin Fadaeinejad
"""
import argparse
from .base_options import BaseOptions
from options.test_options import TestOptions


class Face2FaceTestOption(BaseOptions):
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
        parser.add_argument('--mode', type=str, default='test', help="Model mode to -> [ train | test ] ")
        """
        The default base options for the pix2pix train model
        """
        parser = TestOptions.initialize(self, parser)
        self.isTrain = False
        return parser
