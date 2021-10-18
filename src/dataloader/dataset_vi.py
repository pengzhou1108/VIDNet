import errno
import hashlib
import os
import os.path as osp
import sys
import tarfile
import h5py
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import time
from scipy.misc import imresize
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import random
from .transforms.transforms import Affine
import glob
import json
from .dataset_utils import readFlow,ela,separable_mf, random_noise, random_jpeg
from args import get_parser
import random 
import cv2

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()

if args.dataset == 'youtube':
    from misc.config_youtubeVOS import cfg as cfg_youtube
else:
    from misc.config import cfg


class MyDataset(data.Dataset):

    def __init__(self,
                 args,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split = 'train',
                 resize = False,
                 inputRes = None,
                 video_mode = True,
                 use_prev_mask = False,
                 use_ela=False):


        self.max_seq_len = args.gt_maxseqlen
        self._length_clip = args.length_clip
        self.classes = []
        self.augment = augment
        self.split = split
        self.inputRes = inputRes
        self.video_mode = video_mode
        self.dataset = args.dataset
        self.use_prev_mask = use_prev_mask

    def get_classes(self):
        return self.classes

    def get_raw_sample(self,index):
        """
        Returns sample data in raw format (no resize)
        """
        img = []
        ins = []
        seg = []

        return img, ins, seg

        
    #__getitem__ method has been implemented to get a set of consecutive N (self._length_clip) frames from a given sequence and their
    #respective ground truth annotations.
    def __getitem__(self, index):
        if self.video_mode:
            if self.split == 'train' or self.split == 'val' or self.split == 'trainval':

                edict = self.get_raw_sample_clip(index)
                img = edict['images']
                annot = edict['annotations']
                if self.dataset == 'youtube':
                    if self.split == 'train':
                        img_root_dir = cfg_youtube.PATH.SEQUENCES_TRAIN
                        annot_root_dir = cfg_youtube.PATH.ANNOTATIONS_TRAIN
                    elif self.split == 'val':
                        img_root_dir = cfg_youtube.PATH.SEQUENCES_VAL
                        annot_root_dir = cfg_youtube.PATH.ANNOTATIONS_VAL
                    else:
                        img_root_dir = cfg_youtube.PATH.SEQUENCES_TRAINVAL
                        annot_root_dir = cfg_youtube.PATH.ANNOTATIONS_TRAINVAL
                else:
                    if cfg.PATH.SEQUENCES.split('/')[-1] in ['train', 'val', 'trainval']:
                      cfg.PATH.SEQUENCES = '/'.join(cfg.PATH.SEQUENCES.split('/')[:-1])+'/' + self.split
                      cfg.PATH.SEQUENCES2 = '/'.join(cfg.PATH.SEQUENCES2.split('/')[:-1])+'/' + self.split    
                    else:
                      cfg.PATH.SEQUENCES = cfg.PATH.SEQUENCES + '/' + self.split
                      cfg.PATH.SEQUENCES2 = cfg.PATH.SEQUENCES2 + '/' + self.split

                    img_root_dir = cfg.PATH.SEQUENCES
                    img_ela_dir = '/'.join(img_root_dir.split('/')[:-1])+'/ela'
                    img1_ela_dir = img_ela_dir.replace(cfg.PATH.SEQUENCES, cfg.PATH.SEQUENCES2)

                    annot_root_dir = cfg.PATH.ANNOTATIONS
                    img_original_dir = cfg.PATH.ORIGINAL
                seq_name = img.name
                img_seq_dir = osp.join(img_root_dir, seq_name)
                
                #img_org_dir = osp.join(img_original_dir, seq_name)
                #annot_seq_dir = osp.join(annot_root_dir, annot.name)
                annot_seq_dir = osp.join(annot_root_dir, seq_name)
                starting_frame = img.starting_frame

                imgs = []
                imgs1 = []
                imgs_org = []
                targets = []
                targets_edge = []
                #imgs_flow = []
                imgs_ela = []
                imgs1_ela = []                

                flip_clip = (random.random() < 0.5)
                interp_clip = random.random()
                # Check if img._files are ustrings or strings
                if type(img._files[0]) == str:
                    images = [f for f in img._files]
                else:
                    images = [str(f.decode()) for f in img._files]
                #frame_img = osp.join(img_seq_dir,'%05d.jpg' % starting_frame)
                frame_img = osp.join(img_seq_dir,'%05d.png' % starting_frame)
                starting_frame_idx = images.index(frame_img)

                max_ii = min(self._length_clip,len(images))
            
                #interval = random.randint(1,(len(images)-starting_frame_idx)//5+1)
                for ii in range(max_ii):
                    
                    frame_idx = starting_frame_idx + ii#*interval
                    frame_idx = int(osp.splitext(osp.basename(images[frame_idx]))[0])
                
                    #frame_img = osp.join(img_seq_dir,'%05d.jpg' % frame_idx)
                    #try:
                    frame_img = osp.join(img_seq_dir,'%05d.png' % frame_idx)

                    img = Image.open(frame_img)
                    img1 = Image.open(frame_img.replace(cfg.PATH.SEQUENCES,cfg.PATH.SEQUENCES2))
                    img_org = Image.open(frame_img.replace(cfg.PATH.SEQUENCES,cfg.PATH.ORIGINAL).replace('.png','.jpg'))
                    #img_flow = readFlow(frame_img.replace(cfg.PATH.SEQUENCES, cfg.PATH.FLOW).replace('png','flo'))
                    
                    frame_annot = osp.join(annot_seq_dir,'%05d.png' % frame_idx)
                    annot = Image.open(frame_annot).convert('L')
                    #except:
                        #print(frame_img, frame_annot)
                    img_ela = Image.open(frame_img.replace(cfg.PATH.SEQUENCES,img_ela_dir))
                    img1_ela = Image.open(frame_img.replace(cfg.PATH.SEQUENCES,img1_ela_dir))



                    if self.inputRes is not None:
                        img = imresize(img, self.inputRes)
                        img1 = imresize(img1, self.inputRes)
                        img_org = imresize(img_org, self.inputRes)
                        #img_flow = np.transpose(np.resize(img_flow, (self.inputRes[0],self.inputRes[1],2)),(2,0,1))
                        #print(frame_annot, annot.size)
                        #annot = imresize(annot, self.inputRes, interp='nearest')
                        #annot = imresize(np.array(annot), self.inputRes)
                        annot = np.array(annot.resize((self.inputRes[1],self.inputRes[0])))

                        img_ela = imresize(img_ela, self.inputRes)
                        img1_ela = imresize(img1_ela, self.inputRes)                        

                    #img_ela = np.zeros(img.shape)
                    #img1_ela = np.zeros(img1.shape)

                    #img_ela = separable_mf(img)
                    #img1_ela = separable_mf(img1)
                    if False and interp_clip<0.5:
                        img = random_noise(img)
                        img1 = random_noise(img1)
                        img_ela = random_noise(img_ela)
                        img1_ela = random_noise(img1_ela)                       
                    if False and interp_clip<0.5:
                        img = random_jpeg(img)
                        img1 = random_jpeg(img1,name='b')
                        img_ela = random_jpeg(img_ela,name='c')
                        img1_ela = random_jpeg(img1_ela,name='d')
                        

                    if self.transform is not None:
                        # involves transform from PIL to tensor and mean and std normalization
                        img = self.transform(img)
                        img1 = self.transform(img1)                       
                        img_org = self.transform(img_org)

                        img_ela = self.transform(img_ela)
                        img1_ela = self.transform(img1_ela)                    
                    annot = np.expand_dims(annot, axis=0)


                    if flip_clip and self.split=='train': #and self.flip:
                        img = np.flip(img.numpy(),axis=2).copy()
                        img = torch.from_numpy(img)
                        img1 = np.flip(img1.numpy(),axis=2).copy()
                        img1 = torch.from_numpy(img1)

                        img_ela = np.flip(img_ela.numpy(),axis=2).copy()
                        img_ela = torch.from_numpy(img_ela)
                        img1_ela = np.flip(img1_ela.numpy(),axis=2).copy()
                        img1_ela = torch.from_numpy(img1_ela)

                        img_org = np.flip(img_org.numpy(),axis=2).copy()
                        img_org = torch.from_numpy(img_org)                        
                        annot = np.flip(annot,axis=2).copy()
                        #img_flow = np.flip(img_flow,axis=2).copy()
                        #img_flow = torch.from_numpy(img_flow)

                    if True and random.random()<0.5 and self.split=='train':
                        img = (img.numpy()*interp_clip+(1-interp_clip)*img1.numpy()).copy()
                        img = torch.from_numpy(img)
                        img1 = (img.numpy()*(1-interp_clip)+interp_clip*img1.numpy()).copy()
                        img1 = torch.from_numpy(img1)

                        img_ela = (img_ela.numpy()*interp_clip+(1-interp_clip)*img1_ela.numpy()).copy()
                        img_ela = torch.from_numpy(img_ela)
                        img1_ela = (img_ela.numpy()*(1-interp_clip)+interp_clip*img1_ela.numpy()).copy()
                        img1_ela = torch.from_numpy(img1_ela)    

                    #edge_annot = binary_dilation(annot[0],structure=np.ones((5,5))).astype(annot.dtype) - binary_erosion(annot[0],structure=np.ones((5,5))).astype(annot.dtype)
                    annot = torch.from_numpy(annot)
                    annot = annot.float()

                    #edge_annot = torch.from_numpy((edge_annot)[np.newaxis,:,:])
                    #edge_annot = edge_annot.float()        
                    if self.augmentation_transform is not None and self._length_clip == 1:
                        img, annot= self.augmentation_transform(img, annot)

                    elif self.augmentation_transform is not None and self._length_clip > 1 and ii == 0:
                        tf_matrix = self.augmentation_transform(img)
                        tf_function = Affine(tf_matrix,interp='nearest')
                        #img, annot = tf_function(img,annot)
                        img, annot, img1, img_org = tf_function(img,annot,img1,img_org,img_flow)
                    elif self.augmentation_transform is not None and self._length_clip > 1 and ii > 0:
                        #try:
                        #img, annot = tf_function(img,annot)
                        img, annot, img1, img_org = tf_function(img,annot,img1,img_org,img_flow)
                        #except:
                            #print(annot.shape, frame_annot)
                                            
                    annot = annot.numpy().squeeze() 

                    target = self.sequence_from_masks(seq_name,annot)

                    #edge_annot = edge_annot.numpy().squeeze()   
                    #target_edge = self.sequence_from_masks(seq_name,edge_annot)
                    if self.target_transform is not None:
                        target = self.target_transform(target)
                    
                    imgs.append(img)
                    imgs1.append(img1)
                    imgs_org.append(img_org)
                    targets.append(target)
                    #targets_edge.append(target_edge)
                    imgs_flow = []
                    #imgs_flow.append(img_flow)
                    imgs_ela.append(img_ela)
                    imgs1_ela.append(img1_ela)                    

                                    
                return imgs, imgs1, imgs_org, targets, seq_name, starting_frame, imgs_flow, imgs_ela, imgs1_ela
                #return imgs, targets, seq_name, starting_frame
            else:
                edict = self.get_raw_sample_clip(index)
                img = edict['images']
                if self.dataset == 'youtube':
                    img_root_dir = cfg_youtube.PATH.SEQUENCES_TEST
                else:
                    img_root_dir = cfg.PATH.SEQUENCES
                        
                img_seq_dir = osp.join(img_root_dir, img.name)
                
                starting_frame = img.starting_frame
                seq_name = img.name

                imgs = []                
                images = glob.glob(osp.join(img_seq_dir,'*.jpg'))
                images.sort()
                frame_img = osp.join(img_seq_dir,'%05d.jpg' % starting_frame)
                starting_frame_idx = images.index(frame_img)
                
                max_ii = min(self._length_clip,len(images)-starting_frame_idx)
                
                for ii in range(max_ii):
                    
                    frame_idx = starting_frame_idx + ii
                    frame_idx = int(osp.splitext(osp.basename(images[frame_idx]))[0])
                
                    frame_img = osp.join(img_seq_dir,'%05d.jpg' % frame_idx)
                    img = Image.open(frame_img)

                    if self.inputRes is not None:
                        img = imresize(img, self.inputRes)

                    if self.transform is not None:
                        # involves transform from PIL to tensor and mean and std normalization
                        img = self.transform(img)                    
                    
                    imgs.append(img)

                return imgs, seq_name, starting_frame

    def __len__(self):
        if self.video_mode:
            return len(self.sequence_clips)
        else:
            return len(self.image_files)

    def get_sample_list(self):
        if self.video_mode:
            return self.sequence_clips
        else:
            return self.image_files
        
    def sequence_from_masks(self, seq_name, annot):
        """
        Reads segmentation masks and outputs sequence of binary masks and labels
        """

        if self.dataset == 'youtube':
            if self.split == 'train':
                json_data = open(cfg_youtube.FILES.DB_INFO_TRAIN)
            elif self.split == 'val':
                json_data = open(cfg_youtube.FILES.DB_INFO_VAL)
            else:
                json_data = open(cfg_youtube.FILES.DB_INFO_TRAINVAL)

            data = json.load(json_data)
            instance_ids_str = data['videos'][seq_name]['objects'].keys()
            instance_ids = []
            for id in instance_ids_str:
                instance_ids.append(int(id))
        else:
            instance_ids = np.unique(annot)[1:]
            #print(instance_ids)
            #In DAVIS 2017, some objects not present in the initial frame are annotated in some future frames with ID 255. We discard any id with value 255.
            #if len(instance_ids) > 0:
                    #instance_ids = instance_ids[:-1] if instance_ids[-1]==255 else instance_ids

        h = annot.shape[0]
        w = annot.shape[1]

        total_num_instances = len(instance_ids)
        max_instance_id = 0
        if total_num_instances > 0:
            #max_instance_id = int(np.max(instance_ids))
            max_instance_id = 1
        num_instances = max(self.max_seq_len,max_instance_id)

        gt_seg = np.zeros((num_instances, h*w))
        size_masks = np.zeros((num_instances,)) # for sorting by size
        sample_weights_mask = np.zeros((num_instances,1))
        #print(total_num_instances)
        for i in range(total_num_instances):

            id_instance = int(instance_ids[i])
            aux_mask = np.zeros((h, w))
            aux_mask[annot==id_instance] = 1
            #gt_seg[id_instance-1,:] = np.reshape(aux_mask,h*w)
            #size_masks[id_instance-1] = np.sum(gt_seg[id_instance-1,:])
            #sample_weights_mask[id_instance-1] = 1
            gt_seg[i,:] = np.reshape(aux_mask,h*w)
            size_masks[i] = np.sum(gt_seg[i,:])
            sample_weights_mask[i] = 1
        gt_seg = gt_seg[:][:self.max_seq_len]
        sample_weights_mask = sample_weights_mask[:][:self.max_seq_len]

        targets = np.concatenate((gt_seg,sample_weights_mask),axis=1)

        return targets
