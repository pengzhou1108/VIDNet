import numpy as np 
import os
import pdb
import cv2
import argparse
import glob
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', metavar='INPUT', 
                        help='filenames of input images', required=True)

    parser.add_argument('--dataset', '-o', metavar='INPUT', 
                        help='filenames of ouput images')
    parser.add_argument('--mask_dir', '-d', metavar='mask', 
                        help='filenames of ouput images')
    parser.add_argument('--im_dir', '-l', metavar='mask', 
                        help='filenames of ouput images')
    return parser.parse_args()
def iou_score(output, target):
    smooth = 1e-5

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)\

if __name__ == "__main__":
    args = get_args()

    in_files = glob.glob(os.path.join(args.im_dir,'*/*.png'))
    if in_files==[]:
        in_files = glob.glob(os.path.join(args.im_dir,'*/*.jpg'))
    F1 = []
    F1_1 = []
    IoU = []
    #pdb.set_trace()
    #out_files = get_output_filenames(args, in_files)
    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
        file_name = os.path.splitext(fn.split('/')[-1])[0]
        mask_file = os.path.join(args.mask_dir,fn.split('/')[-2],file_name+'.png')
        im_mask_lr =  scipy.misc.imread(mask_file, 'L')
        try:
            img = scipy.misc.imread(os.path.join(args.im_dir,fn.split('/')[-2],file_name+'.png'))
        except:
            img = scipy.misc.imread(os.path.join(args.im_dir,fn.split('/')[-2],file_name+'.jpg'))
        max_val = im_mask_lr.max()
        im_mask_lr = cv2.resize(im_mask_lr.astype(np.uint8),(img.shape[1], img.shape[0]))
        #pdb.set_trace()
        #im_mask_lr = cv2.resize(im_mask_lr.astype(np.uint8),(im_mask_hr.shape[1], im_mask_hr.shape[0]))
        #gt_mask = (cv2.Laplacian((im_mask_hr>128).astype(np.uint8),cv2.CV_8U)>0).astype(int)
        gt_mask = im_mask_lr>(max_val/2)
        result = scipy.misc.imread(os.path.join(args.input,fn.split('/')[-2],file_name+'_instance_00.png'), 'L')
        result = cv2.resize(result,(img.shape[1], img.shape[0]))/255
        recall = np.sum(gt_mask*(np.array(result)>0.5))/np.sum(gt_mask)
        precision = np.sum(gt_mask*(np.array(result)>0.5))/(np.sum(np.array(result)>0.5)+1e-6)
        f1 = 2*(precision*recall)/(precision+recall+1e-6)
        F1_1.append(f1)
        iou = iou_score(result, gt_mask)
        IoU.append(iou)
        print(np.mean(F1_1),np.mean(IoU))
    print('average F1:{}, IoU:{}'.format(np.mean(F1_1), np.mean(IoU)))
