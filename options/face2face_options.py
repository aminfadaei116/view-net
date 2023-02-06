# -*- coding: utf-8 -*-
"""
This script contains the options used doe the face2face model

@author: Amin Fadaeinejad
"""
import argparse
from .base_options import BaseOptions
from options.test_options import TestOptions
from options.train_options import TrainOptions


class Face2FaceOption(BaseOptions):
    """
    This class defines options used for the face2face model
    """
    def __init__(self) -> None:
        """
        The first three are for the face2face model itself
        """
        parser = argparse.ArgumentParser(description='Gather the parameters for the project')
        parser.add_argument('--used_device', type=str, default='none',
                            help='Which device are you using at the moment-> '
                                 '[ YorkU | Ubisoft | laptop ]')
        parser.add_argument('--mode', type=str, default='test', help="Model mode to -> [ train | test ] ")
        """
        The default base options for the pix2pix model
        """
        # parser = BaseOptions.initialize(self, parser)
        """
        The options for test or tran mode
        """
        if parser.parse_args().mode == "train":
            parser = TrainOptions.initialize(self, parser)
        elif parser.parse_args().mode == "test":
            parser = TestOptions.initialize(self, parser)
        else:
            raise Exception("Mode not valid")
        self.parser = parser

    def get_parser(self):
        """
        get the parser parameters
        :return: class parser
            Class containing the parameter information
        """
        return self.parser.parse_args()
