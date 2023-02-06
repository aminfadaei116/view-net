# -*- coding: utf-8 -*-
"""
This script contains the options used doe the face2face model

@author: Amin Fadaeinejad
"""
import argparse


class Face2FaceOption:
    """
    This class defines options used for the face2face model
    """
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description='Gather the parameters for the project')
        parser.add_argument('--used_device', type=str, default='none',
                            help='Which device are you using at the moment-> '
                                 '[ YorkU | Ubisoft | laptop ]')
        self.parser = parser

    def get_parser(self):
        """
        get the parser parameters
        :return: class parser
            Class containing the parameter information
        """
        return self.parser.parse_args()
