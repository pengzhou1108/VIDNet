import sys, os.path
from PIL import Image, ImageChops, ImageEnhance
import pdb
import numpy as np
import cv2
import glob
import os
from scipy.ndimage import gaussian_filter,median_filter
def ela(im):
    resaved = 'resaved.jpg'
    #resaved1 = 'resaved1.jpg'
    ela = 'ela.png'

    #im.save(resaved1, 'JPEG', quality=90)
    im.save(resaved, 'JPEG', quality=50)
    resaved_im = Image.open(resaved)
    #resaved_im1 = Image.open(resaved1)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff
    #pdb.set_trace()
    #ela_im = Image.fromarray(abs(np.array(ela_im)-np.array(ela_im).mean()))
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    return ela_im

if __name__ == "__main__":
    davis_train = glob.glob('/vulcan/scratch/pengzhou/model/opn-demo/vi_davis/train/*/*.png')
    davis_test = glob.glob('/vulcan/scratch/pengzhou/model/opn-demo/vi_davis/val/*/*.png')
    out_dir ='/vulcan/scratch/pengzhou/model/opn-demo/vi_davis/ela'
    for file in davis_train+davis_test:

        im = Image.open(file)
        #im2 = Image.open('./00012.png')
        #ind = np.where(np.array(im2)>0)
        #aa = Image.fromarray(np.array(im)[ind[0].min():ind[0].max(),ind[1].min():ind[1].max(),:])
        ela_im = ela(im)
        #pdb.set_trace()
        #aa.save('b.png')
        if not os.path.exists(os.path.join(out_dir, file.split('/')[-2])):
            os.makedirs(os.path.join(out_dir, file.split('/')[-2]))

        ela_im.save(os.path.join(out_dir, file.split('/')[-2],file.split('/')[-1]))