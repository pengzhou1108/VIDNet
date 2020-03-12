import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage.interpolation import zoom
import torch
from PIL import Image, ImageChops, ImageEnhance
import random
import cv2
def separable_mf(im):
    im_array = np.array(im)
    return np.stack([abs(im_array[:,:,i] - median_filter(im_array[:,:,i],size=5)) for i in range(3)],-1)
def random_jpeg(img,quality=90,name='a'):
    img = Image.fromarray(img.astype(np.uint8))
    img.save('{:}.jpg'.format(name),'JPEG', quality=quality)
    #pdb.set_trace()
    img=Image.open('{:}.jpg'.format(name))
    return img
def random_noise(img,dB=20.0):
    img = np.array(img)
    row,col,ch= img.shape
    P_sig=np.mean(img)
    #sigma = np.sqrt(10**(dB/10))
    sigma = P_sig/(10**(dB/10))
    gauss = np.random.normal(0,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    img = img + gauss
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    img = Image.fromarray(img.astype(np.uint8))
    return img
def ela(im,name='resaved'):
    resaved = '/vulcan/scratch/pengzhou/model/rvos/scripts/{}.jpg'.format(name)
    ela = 'ela.png'


    im.save(resaved, 'JPEG', quality=50)
    resaved_im = Image.open(resaved)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    del resaved_im
    return ela_im

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))
            
def get_dataset(args, split, image_transforms = None, target_transforms = None, augment = False,inputRes = None, video_mode = True, use_prev_mask = False,use_ela=False):


    if args.dataset =='davis2017' or args.dataset =='davis2016':
        from .davis2017 import DAVISLoader as MyChosenDataset
    elif args.dataset == 'youtube':
        from .youtubeVOS import YoutubeVOSLoader as MyChosenDataset
    elif args.dataset =='davis2016_vi':
        from .davis2017_vi import DAVISLoader as MyChosenDataset
    
    


    dataset = MyChosenDataset(args,
                            split = split,
                            transform = image_transforms,
                            target_transform = target_transforms,
                            augment = augment,
                            resize = args.resize,
                            inputRes = inputRes,
                            video_mode = video_mode,
                            use_prev_mask = use_prev_mask,
                            use_ela=use_ela)
    return dataset
    
def sequence_palette():

    # RGB to int conversion

    palette = {(  0,   0,   0) : 0 ,
             (0,   255,   0) : 1 ,
             (  255, 0,   0) : 2 ,
             (0, 0,   255) : 3 ,
             (  255,   0, 255) : 4 ,
             (0,   255, 255) : 5 ,
             (  255, 128, 0) : 6 ,
             (102, 0, 102) : 7 ,
             ( 51,   153,   255) : 8 ,
             (153,   153,   255) : 9 ,
             ( 153, 153,   0) : 10,
             (178, 102,   255) : 11,
             ( 204,   0, 204) : 12,
             (0,   102, 0) : 13,
             ( 102, 0, 0) : 14,
             (51, 0, 0) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20,
             (224,  224, 192) : 21 }

    return palette
