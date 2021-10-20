import sys, os.path
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import glob
import os
from scipy.ndimage import gaussian_filter,median_filter
"""
This code is used to extract the ELA frames from the original RGB frames
"""
def ela(im):
    resaved = 'resaved.jpg'
    ela = 'ela.png'

    im.save(resaved, 'JPEG', quality=50)
    resaved_im = Image.open(resaved)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    return ela_im

if __name__ == "__main__":
    davis_train = glob.glob('../../model/opn-demo/vi_davis/train/*/*.png') # FIXME
    davis_test = glob.glob('../../model/opn-demo/vi_davis/val/*/*.png')# FIXME
    out_dir ='../../model/opn-demo/vi_davis/ela'
    for file in davis_train+davis_test:

        im = Image.open(file)
        ela_im = ela(im)
        if not os.path.exists(os.path.join(out_dir, file.split('/')[-2])):
            os.makedirs(os.path.join(out_dir, file.split('/')[-2]))

        ela_im.save(os.path.join(out_dir, file.split('/')[-2],file.split('/')[-1]))