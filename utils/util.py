# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:49:28 2022

This script contains the functions of our model

@author: Amin Fadaeinejad
"""
import matplotlib.pyplot as plt
import torch


def create_mask(config, keys, height, width, img=None) -> torch.tensor:
    """
    Converts the keypoints location to a mask marked by those keypoints.
    If img is passed, the keypoints are going to be added to the image
    :param:
        config: class Config
        keys: torch.tensor
        height: int
        width: int
        img: torch.tensor
    :return:
        torch.tensor
    """
    if img is not None:
        new_img = img.clone().detach().to(config.DEVICE)
        new_img[:, (height * keys[:, 1]).long(), (width * keys[:, 0]).long()] = 255.0
        return new_img
    else:
        mask = torch.zeros((height, width), device=config.DEVICE)
        mask[(height * keys[:, 1]).long(), (width * keys[:, 0]).long()] = 1
        return mask


def create_mask2(config, keys, height, width, img=None) -> torch.tensor:
    """
    Converts the keypoints location to a mask marked by those keypoints.
    If img is passed, the keypoints are going to be added to the image.
    The difference between this and create_mask is the size of the keypoints on the mask
    :param:
        config: class Config
        keys: torch.tensor
        height: int
        width: int
        img: torch.tensor
    :return:
        torch.tensor
    """
    if img is not None:
        new_img = img.clone().detach().to(config.DEVICE)
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_img[:, (height * keys[:, 1]).long() + i, (width * keys[:, 0]).long() + j] = 255.0
        return new_img
    else:
        mask = torch.zeros((height, width), device=config.DEVICE)
        for i in range(-1, 2):
            for j in range(-1, 2):
                mask[(height * keys[:, 1]).long() + i, (width * keys[:, 0]).long() + j] = 1
        return mask


def show_image_tensor(img, is3chan=True, is_output=False, return_output=False) -> torch.tensor:
    """
    Converts a multi dimensional tensor to an image
    :param:
        img: torch.tensor
        is3chan: bool
        is_output: bool
    :return:
        torch.tensor
    """
    new_img = img.clone().detach()
    if is_output:
        new_img = torch.squeeze(new_img)

    new_img = new_img.to('cpu')
    if is3chan:
        plt.imshow((new_img.permute(1, 2, 0)))
        plt.xlabel("x axis")
        plt.ylabel("y axis")
    else:
        plt.imshow(new_img)
        plt.xlabel("x axis")
        plt.ylabel("y axis")
    if return_output:
        return new_img


def draw(width, height, ref_key, tar_key, size=10, connect=False) -> None:
    """
    Will draw two sets of keypoints, can also connect the corresponding keypoints to each other
    :param:
        width: int
        height: int
        ref_key: torch.tensor
        tar_key: torch.tensor
        size: int
        connect: bool
    :return:
        None
    """
    key1 = torch.zeros((len(ref_key), 2))
    key2 = torch.zeros((len(tar_key), 2))
    key1[:, 0] = ref_key[:, 0]
    key1[:, 1] = tar_key[:, 0]

    key2[:, 0] = ref_key[:, 1]
    key2[:, 1] = tar_key[:, 1]

    if connect:
        for i in range(len(key1)):
            plt.plot(key1[i].cpu().numpy() * width, key2[i].cpu().numpy() * height)

    plt.scatter(ref_key[:, 0].cpu().numpy() * width, ref_key[:, 1].cpu().numpy() * height, s=size)
    plt.scatter(tar_key[:, 0].cpu().numpy() * width, tar_key[:, 1].cpu().numpy() * height, s=size)

    plt.xlim(0, width - 1)
    plt.ylim(0, height - 1)
    plt.xlabel("x axis")
    plt.ylabel("y axis")

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()


def transform_keys(config, keys, euler, translation) -> torch.tensor:
    """
    Applies a transformation, which consists of a rational and a translation
    :param:
        config: class Config
        keys: torch.tensor
        euler: torch.tensor [3,]
        translation: torch.tensor [3,1]
    :return:
        torch.tensor
    """
    rotation_matrix = torch.linalg.multi_dot((r_x(config, euler[0]), r_y(config, euler[1]), r_z(config, euler[2])))
    center = torch.mean(keys, dim=0)
    keys = keys - center
    out = torch.matmul(keys, rotation_matrix)
    return out + center + translation


def r_x(config, theta) -> torch.tensor:
    """
    Rotation matrix over the x-axis
    :param:
        config: class Config
        theta: torch.tensor [1,]
    :return:
        torch.tensor
    """
    return torch.tensor([[1, 0, 0],
                         [0, torch.cos(theta), -torch.sin(theta)],
                         [0, torch.sin(theta), torch.cos(theta)]], device=config.DEVICE, dtype=torch.double)


def r_y(config, theta) -> torch.tensor:
    """
    Rotation matrix over the y-axis
    :param:
        config: class Config
        theta: torch.tensor [1,]
    :return:
        torch.tensor
    """
    return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                         [0, 1, 0],
                         [-torch.sin(theta), 0, torch.cos(theta)]], device=config.DEVICE, dtype=torch.double)


def r_z(config, theta) -> torch.tensor:
    """
    Rotation matrix over the z-axis
    :param:
        config: class Config
        theta: torch.tensor [1,]
    :return:
        torch.tensor
    """
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0],
                         [0, 0, 1]], device=config.DEVICE, dtype=torch.double)
