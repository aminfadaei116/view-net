# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:55:15 2022

This script contains the hyper parameters of our model

@author: Amin Fadaeinejad
"""

import torch
import mediapipe as mp


class Config:

    def __init__(self, system):
        self.system = system
        if self.system == "Ubisoft":
            username = "afadaeinejad/OneDrive - Ubisoft"
        elif self.system == "YorkU":
            username = "afadaei"
        elif self.system == "laptop":
            username = "Amin"

        self.PathNPY1 = f"C:/Users/{username}/Documents/GitHub/view-gen/FaceData/Person_2/Image_120.npy"
        self.PathNPY2 = f"C:/Users/{username}/Documents/GitHub/view-gen/FaceData/Person_2/Image_34.npy"

        self.PathImg1 = f"C:/Users/{username}/Documents/GitHub/view-gen/FaceData/Person_2/Image_120.jpg"
        self.PathImg2 = f"C:/Users/{username}/Documents/GitHub/view-gen/FaceData/Person_2/Image_34.jpg"

        self.path = f'C:/Users/{username}/Documents/GitHub/view-gen/FaceData/Person_7'

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.NUM_CHANNALS = 3
        self.FACE_LANKMARK_LENGTH = 478
        self.NUMBER_COORDINATE = 3
        self.LOAD_MODEL = False
        self.LEARNING_RATE = 0.001
        self.pi = 3.14159265359