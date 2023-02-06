# -*- coding: utf-8 -*-
"""
This script contains the hyper parameters of our model

@author: Amin Fadaeinejad
"""

import torch
import mediapipe as mp


class Config:
    """

    A class used to represent the project configuration

    example: --used_device YorkU

    """
    def __init__(self, parser) -> None:
        """
        Creating the configuration class
        :param:
            parser: str [Ubisoft | YorkU | laptop]
                The device that I am currently using
        """
        if parser.used_device == "Ubisoft":
            username = "afadaeinejad/OneDrive - Ubisoft"
        elif parser.used_device == "YorkU":
            username = "afadaei"
        elif parser.used_device == "laptop":
            username = "Amin"
        else:
            raise Exception("System not detected (--used_device no valid)")

        self.PathNPY1 = f"C:/Users/{username}/Documents/GitHub/view-gen/face-data/person_02/Image_120.npy"
        self.PathNPY2 = f"C:/Users/{username}/Documents/GitHub/view-gen/face-data/person_02/Image_34.npy"

        self.PathImg1 = f"C:/Users/{username}/Documents/GitHub/view-gen/face-data/person_02/Image_120.jpg"
        self.PathImg2 = f"C:/Users/{username}/Documents/GitHub/view-gen/face-data/person_02/Image_34.jpg"

        self.path = f'C:/Users/{username}/Documents/GitHub/view-gen/face-data/person_07'
        self.used_device = parser.used_device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.num_channel = 3
        self.landmark_length = 478
        self.num_coordinate = 3
        self.load_model = False
        self.learning_rate = 0.001
        self.pi = 3.14159265359
