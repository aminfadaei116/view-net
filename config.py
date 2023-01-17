# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:55:15 2022

This script contains the hyper parameters of our model

@author: Amin Fadaeinejad
"""

import torch
import mediapipe as mp

PathNPY1 = r"C:\Users\afadaei\Documents\GitHub\ViewGen\FaceData\Person_2\Image_120.npy"
PathNPY2 = r"C:\Users\afadaei\Documents\GitHub\ViewGen\FaceData\Person_2\Image_34.npy"

PathImg1 = r"C:\Users\afadaei\Documents\GitHub\ViewGen\FaceData\Person_2\Image_120.jpg"
PathImg2 = r"C:\Users\afadaei\Documents\GitHub\ViewGen\FaceData\Person_2\Image_34.jpg"

path = r'C:\Users\afadaei\Documents\GitHub\ViewGen\FaceData\Person_7'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
NUM_CHANNALS = 3
FACE_LANKMARK_LENGTH = 478
NUMBER_COORDINATE = 3


pi = 3.14159265359