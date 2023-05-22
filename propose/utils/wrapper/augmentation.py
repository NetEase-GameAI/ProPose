import math
import random
import numpy as np


def add_occlusion(src, xmin, xmax, ymin, ymax, imgwidth, imgheight):
    while True:
        area_min = 0.0
        area_max = 0.2
        synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

        ratio_min = 0.5
        ratio_max = 1.5
        synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

        synth_h = math.sqrt(synth_area * synth_ratio)
        synth_w = math.sqrt(synth_area / synth_ratio)
        synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
        synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

        if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imgheight:
            synth_xmin = int(synth_xmin)
            synth_ymin = int(synth_ymin)
            synth_w = int(synth_w)
            synth_h = int(synth_h)
            src[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
            break


def add_DPG(bbox, imgwidth, imght):
    """ Add dpg for data augmentation, including random crop and random sample. """
    PatchScale = random.uniform(0, 1)
    width = bbox[2] - bbox[0]
    ht = bbox[3] - bbox[1]

    if PatchScale > 0.85:
        ratio = ht / width
        if (width < ht):
            patchWidth = PatchScale * width
            patchHt = patchWidth * ratio
        else:
            patchHt = PatchScale * ht
            patchWidth = patchHt / ratio

        xmin = bbox[0] + random.uniform(0, 1) * (width - patchWidth)
        ymin = bbox[1] + random.uniform(0, 1) * (ht - patchHt)
        xmax = xmin + patchWidth + 1
        ymax = ymin + patchHt + 1
    else:
        xmin = max(
            1, min(bbox[0] + np.random.normal(-0.0142, 0.1158) * width, imgwidth - 3))
        ymin = max(
            1, min(bbox[1] + np.random.normal(0.0043, 0.068) * ht, imght - 3))
        xmax = min(
            max(xmin + 2, bbox[2] + np.random.normal(0.0154, 0.1337) * width), imgwidth - 3)
        ymax = min(
            max(ymin + 2, bbox[3] + np.random.normal(-0.0013, 0.0711) * ht), imght - 3)

    bbox[0] = xmin
    bbox[1] = ymin
    bbox[2] = xmax
    bbox[3] = ymax

    return bbox