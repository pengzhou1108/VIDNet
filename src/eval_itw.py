import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from args import get_parser
from utils.utils import batch_to_var, batch_to_var_test, make_dir, outs_perms_to_cpu, load_checkpoint, check_parallel
from modules.model import RSIS, FeatureExtractor
from test import test,test_ela
from dataloader.dataset_utils import sequence_palette, ela
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from scipy.misc import toimage
#import scipy
import glob
from dataloader.dataset_utils import get_dataset
import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import sys, os
import json
import pdb
from misc.config import cfg
from torch.autograd import Variable
class Sequence:
    def __init__(self, args, seq_name):
        # Frames and annotation paths
        self.frames_path = os.path.join(args.frames_path,seq_name)
        self.init_mask_path = os.path.join(args.init_mask_path,seq_name)
        self.seq_name = seq_name
        # Frames information
        self.frames_list = None
        self.input_res = (240, 427)
        self.max_instances = args.maxseqlen  # Limit the max number of instances
        # Frame and annotation data
        self.imgs_data = []
        self.imgs_ela_data = []
        self.imgnames = []
        self.init_mask_data = None
        self.instance_ids = None
        # Frames normalization
        self.img_transforms = None
        self._generate_transform()
        # Initialize variables
        self._get_frames_list()
        self.load_frames()
        if not args.zero_shot:
            # Semi-supervised
            self.load_annot(args.use_gpu)
        if args.zero_shot:
            self.instance_ids = np.arange(0, 10)  # Get 10 instances for zero-shot

    def _get_frames_list(self):
        self.frames_list = sorted(os.listdir(self.frames_path))

    def load_frame(self, frame_path):
        img = Image.open(frame_path)
        img_ela = ela(img,'b')
        if self.input_res is not None:
            img = imresize(img, self.input_res)
            img_ela = imresize(img_ela, self.input_res)
        if self.img_transforms is not None:
            img = self.img_transforms(img)
            img_ela = self.img_transforms(img_ela)

        return img,img_ela

    def load_frames(self):
        for frame_name in self.frames_list:
            #pdb.set_trace()
            frame_path = os.path.join(self.frames_path, frame_name)
            img, img_ela = self.load_frame(frame_path)
            self.imgs_data.append(img)
            self.imgs_ela_data.append(img_ela)
            self.imgnames.append(os.path.splitext(frame_name)[0])

    def _generate_transform(self):
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.img_transforms = transforms.Compose([to_tensor, normalize])

    def load_annot(self, use_gpu):
        annot = Image.open(self.init_mask_path)
        if self.input_res is not None:
           annot = imresize(annot, self.input_res, interp='nearest')

        # Prepared for DAVIS-like annotations
        annot = np.expand_dims(annot, axis=0)
        annot = torch.from_numpy(annot)
        annot = annot.float()
        annot = annot.numpy().squeeze()
        annot = self.seg_from_annot(annot)

        prev_mask = annot
        prev_mask = np.expand_dims(prev_mask, axis=0)
        prev_mask = torch.from_numpy(prev_mask)
        y_mask = Variable(prev_mask.float(), requires_grad=False)
        if use_gpu:
            y_mask = y_mask.cuda()
        self.init_mask_data = y_mask

    def seg_from_annot(self, annot):
        instance_ids = sorted(np.unique(annot)[1:])

        h = annot.shape[0]
        w = annot.shape[1]

        total_num_instances = len(instance_ids)
        max_instance_id = 0
        if total_num_instances > 0:
            max_instance_id = int(np.max(instance_ids))
        num_instances = max(self.max_instances, max_instance_id)

        gt_seg = np.zeros((num_instances, h * w))

        for i in range(total_num_instances):
            id_instance = int(instance_ids[i])
            aux_mask = np.zeros((h, w))
            aux_mask[annot == id_instance] = 1
            gt_seg[id_instance - 1, :] = np.reshape(aux_mask, h * w)

        self.instance_ids = instance_ids
        gt_seg = gt_seg[:][:self.max_instances]

        return gt_seg

class Evaluate():

    def __init__(self,args):

        self.split = args.eval_split
        self.dataset = args.dataset
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        image_transforms = transforms.Compose([to_tensor,normalize])
        #image_transforms = transforms.Compose([to_tensor])


        #self.loader = Sequence(args, seq_name)

        self.args = args

        print(args.model_name)
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint(args.model_name,args.use_gpu)
        load_args.use_gpu = args.use_gpu
        self.encoder = FeatureExtractor(load_args)
        self.decoder = RSIS(load_args)
        #pdb.set_trace()
        print(load_args)

        if args.ngpus > 1 and args.use_gpu:
            self.decoder = torch.nn.DataParallel(self.decoder,device_ids=range(args.ngpus))
            self.encoder = torch.nn.DataParallel(self.encoder,device_ids=range(args.ngpus))

        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        self.encoder.load_state_dict(encoder_dict)
        to_be_deleted_dec = []
        for k in decoder_dict.keys():
            if 'fc_stop' in k:
                to_be_deleted_dec.append(k)
        for k in to_be_deleted_dec:
            del decoder_dict[k]
        self.decoder.load_state_dict(decoder_dict)

        if args.use_gpu:
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder.eval()
        self.decoder.eval()
        if load_args.length_clip == 1:
            self.video_mode = False
            print('video mode not activated')
        else:
            self.video_mode = True
            print('video mode activated')
 

    def run_eval(self):
        print ("Dataset is %s"%(self.dataset))
        print ("Split is %s"%(self.split))

        if args.overlay_masks:

            colors = []
            palette = sequence_palette()
            inv_palette = {}
            for k, v in palette.items():
                inv_palette[v] = k
            num_colors = len(inv_palette.keys())
            for id_color in range(num_colors):
                if id_color == 0 or id_color == 21:
                    continue
                c = inv_palette[id_color]
                colors.append(c)

        if self.split == 'val':
            
            if args.dataset == 'youtube':

                masks_sep_dir = os.path.join('../models', args.model_name, 'masks_sep_2assess')
                make_dir(masks_sep_dir)
                if args.overlay_masks:
                    results_dir = os.path.join('../models', args.model_name, 'results')
                    make_dir(results_dir)
            
                json_data = open('../../databases/YouTubeVOS/train/train-val-meta.json')
                data = json.load(json_data)
                
            else:

                masks_sep_dir = os.path.join('../models', args.model_name, 'masks_'+args.frames_path.split('/')[-3])
                #masks_sep_dir = os.path.join('../models', args.model_name, 'masks_sep_2assess-davis')
                make_dir(masks_sep_dir)
                if args.overlay_masks:
                    results_dir = os.path.join('../models', args.model_name, 'results-'+args.frames_path.split('/')[-3])
                    make_dir(results_dir)
            #files = glob.glob('/vulcan/scratch/pengzhou/model/Free-Form-Video-Inpainting/test_outputs/epoch_0/test_object_like/*')
            #files = glob.glob('/vulcan/scratch/pengzhou/model/Free-Form-Video-Inpainting/FVI/Test/JPEGImages/*')
            files = glob.glob('/vulcan/scratch/pengzhou/model/rvos/databases/DAVIS2016/JPEGImages/480p/*')
            for seq_names in files:
                #pdb.set_trace()
                seq_name = seq_names.split('/')[-1]
                seq = Sequence(args, seq_name)
                seq_name = [seq_name]
                for ii, (img,img_ela,name) in enumerate(zip(seq.imgs_data,seq.imgs_ela_data,seq.imgnames)):
                    prev_hidden_temporal_list = None
                    #max_ii = min(len(inputs),args.length_clip)
                    #pdb.set_trace()
                    base_dir_masks_sep = masks_sep_dir + '/' + seq_name[0] + '/'
                    make_dir(base_dir_masks_sep)

                    if args.overlay_masks:
                        base_dir = results_dir + '/' + seq_name[0] + '/'
                        make_dir(base_dir)
                    
                    #for ii in range(max_ii):

                    #                x: input images (N consecutive frames from M different sequences)
                    #                y_mask: ground truth annotations (some of them are zeros to have a fixed length in number of object instances)
                    #                sw_mask: this mask indicates which masks from y_mask are valid
                    #x, y_mask, sw_mask = batch_to_var(args, inputs[ii], targets[ii])
                    x = batch_to_var_test(args, img).unsqueeze(0)
                    x_ela = Variable(img_ela,requires_grad=False).cuda().unsqueeze(0)
                    #x_cat = torch.cat([x,x_ela], 1)
                    #x_cat = x_ela
                    print(seq_name[0] + '/' + '%05d' % (ii))
                    
                    #from one frame to the following frame the prev_hidden_temporal_list is updated.
                    #outs, hidden_temporal_list = test_ela(args, self.encoder, self.decoder, x, prev_hidden_temporal_list,x_ela=x_ela)
                    outs, hidden_temporal_list = test(args, self.encoder, self.decoder, x, prev_hidden_temporal_list,x_ela=x_ela)
                    #outs, hidden_temporal_list = test_edge(args, self.encoder, self.decoder, x_cat, prev_hidden_temporal_list)

                    if args.dataset == 'youtube':
                        num_instances = len(data['videos'][seq_name[0]]['objects'])
                    else:
                        num_instances = 1

                    x_tmp = x.data.cpu().numpy()
                    height = x_tmp.shape[-2]
                    width = x_tmp.shape[-1]
                    for t in range(num_instances):
                        mask_pred = (torch.squeeze(outs[0, t, :])).cpu().numpy()
                        mask_pred = np.reshape(mask_pred, (height, width))
                        indxs_instance = np.where(mask_pred > 0.5)
                        mask2assess = np.zeros((height, width))
                        mask2assess[indxs_instance] = 255
                        toimage(mask2assess, cmin=0, cmax=255).save(
                            base_dir_masks_sep + '%05d_instance_%02d.png' % (int(name), t))
                
                    if args.overlay_masks:

                        frame_img = x.data.cpu().numpy()[0,:,:,:].squeeze()
                        frame_img = np.transpose(frame_img, (1,2,0))
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        frame_img = std * frame_img + mean
                        frame_img = np.clip(frame_img, 0, 1)
                        
                        plt.figure();plt.axis('off')
                        plt.figure();plt.axis('off')
                        plt.imshow(frame_img)
                        #pdb.set_trace()
                        for t in range(num_instances):
                            
                            mask_pred = (torch.squeeze(outs[0,t,:])).cpu().numpy()
                            mask_pred = np.reshape(mask_pred, (height, width))
                            ax = plt.gca()
                            tmp_img = np.ones((mask_pred.shape[0], mask_pred.shape[1], 3))
                            color_mask = np.array(colors[t])/255.0
                            for i in range(3):
                                tmp_img[:,:,i] = color_mask[i]
                            ax.imshow(np.dstack( (tmp_img, mask_pred*0.7) ))
                            
                        figname = base_dir + 'frame_%02d.png' %(int(name))
                        plt.savefig(figname,bbox_inches='tight')
                        plt.close()

                    if self.video_mode:
                        prev_hidden_temporal_list = hidden_temporal_list
            


if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    
    gpu_id = args.gpu_id
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    if not args.log_term:
        print ("Eval logs will be saved to:", os.path.join('../models',args.model_name, 'eval.log'))
        #sys.stdout = open(os.path.join('../models',args.model_name, 'eval.log'), 'w')

    E = Evaluate(args)
    E.run_eval()
