import cv2
from PIL import Image
import torch.nn.functional as F
import numpy as np

import torch


def load_img(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((256, 256), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def replace(h, mask, h_ref, ref_mask=None):
    hh, ww = h.shape[2], h.shape[3]
    resize_mask = F.interpolate(mask, [hh, ww], mode='bicubic')
    return h * resize_mask + h_ref * (1 - resize_mask)


def visualizeGradHeatmap(grad, i, t):
    offset = grad.clone().mean(dim=0, keepdim=True)
    offset = (offset - offset.min()) / (offset.max() - offset.min())
    offset = offset.permute(1, 2, 0).cpu().numpy().squeeze() * 255
    offset = (offset - offset.min()) / (offset.max() - offset.min()) * 255

    offset = offset.astype(np.uint8)
    offset = cv2.resize(offset, (64, 64))

    w = cv2.applyColorMap(offset, cv2.COLORMAP_JET)

    cv2.imwrite('outputs/map/%d_%d_%d.png' % (t, grad.shape[1], i), w)
