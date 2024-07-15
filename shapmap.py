import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


def shap_mapping(g_ema, image, sample, input_is_latent=False, threshold=1):
    mapping = torch.zeros([256, 3, 32, 32])
    img_gen, _ = g_ema([sample], input_is_latent=input_is_latent, noise=None, randomize_noise=False)

    for j in tqdm(range(32)):
        for i in range(3):
            for k in range(32):
                pixel_value = img_gen[0, i, j, k].mean()
                pixel_value.backward(retain_graph=True)
                for l in range(256):
                    mapping[l, i, j, k] = sample.grad[0][l].clone()
                sample.grad.zero_()

    image_np = np.array(image)
    image_tensor = image_np / 255
    image_tensor = np.transpose(image_tensor, (2, 0, 1))
    image_tensor = np.expand_dims(image_tensor, axis=0)
    image_tensor = torch.from_numpy(image_tensor)

    above_threshold = mapping > threshold

    pool_radius = 1
    kernel_size = 1 * pool_radius + 1
    max_pool = torch.nn.MaxPool2d(kernel_size, padding=pool_radius)
    activated_map = max_pool(above_threshold.float())
    activated_map = F.interpolate(activated_map, size=(32, 32), mode='bilinear', align_corners=False)

    map = activated_map.float()
    return map
