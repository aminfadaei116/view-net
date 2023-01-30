# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:49:28 2022

This script contains the functions of our model

@author: Amin Fadaeinejad
"""
import numpy as np
import cv2
from models.image2image import render_image
import torch
from configs.config import Config


def use_webcam(config, height, width, ref_key, img_ref) -> None:
    """
    Generate a new image with the indenity of the img_ref/ref_key and the motion/location of the driving video from the
    webcam
    :param:
      config: class config
        A class that has the configuration parameters
      height: int
      width: int
      ref_key: torch.tensor
      img_ref: torch.tensor
    return:
      None
    """
    cap = cv2.VideoCapture(0)
    with config.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            if config.system == "Ubisoft":
                image = cv2.flip(image, 1)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                land_marks = np.zeros((config.landmark_length, 3))
                for i, land in enumerate(results.multi_face_landmarks[0].landmark):
                    land_marks[i] = [land.x, land.y, land.z]

                land_ten = torch.tensor(land_marks, device=config.DEVICE, requires_grad=False)
                ##
                with torch.no_grad():
                    output = render_image(config, height, width, ref_key, land_ten, img_ref, sd=0.01)

                dummy = torch.squeeze(output)
                # dummy = createMask(landTen, height, width, dummy).int()
                image = (dummy.cpu().permute(1, 2, 0).numpy()).astype(np.uint8)
                # img = showImageTensor(dummy, is3chan=True, isOutput=True, returnOutput=True)
            # showImageTensor(output.int(), is3chan=True, isOutput=True)
            # showImageTensor(output, isOutput=True)

            # cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imshow('MediaPipe Face Mesh', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
