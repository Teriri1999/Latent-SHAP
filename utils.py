import cv2
import numpy as np
import torch


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


def generate_color_heatmap(img):
    img_sqrt = img
    normalized_img = (img_sqrt - img_sqrt.min()) / (img_sqrt.max() - img_sqrt.min())

    colormap = cv2.applyColorMap((normalized_img * 255).astype(np.uint8), cv2.COLORMAP_JET)

    return colormap


def generate_heatmap(img):
    img_sqrt = img
    normalized_img = (img_sqrt - img_sqrt.max()) / (img_sqrt.max() - img_sqrt.min())
    heatmap = (normalized_img * 255).astype(np.uint8)

    return heatmap

