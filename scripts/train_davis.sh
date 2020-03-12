#!/bin/bash

##SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
##SBATCH --qos=default
##SBATCH --partition=dpart
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#export CUDA_VISIBLE_DEVICES=0
export PATH="/vulcan/scratch/pengzhou/tools/anaconda3/bin:${PATH}"
export PYTHONPATH="/vulcan/scratch/pengzhou/tools/anaconda3/lib/python3.6/site-packages:${PYTHONPATH}"
export LD_LIBRARY_PATH="/vulcan/scratch/pengzhou/tools/anaconda3/lib/:${LD_LIBRARY_PATH}"

#python3 ../src/train_vi_edge.py -model_name=davis2016_vgg_2dataset_3clip_raw_in_edge_recursive -dataset=davis2016_vi -batch_size=4 -length_clip=3 -base_model=vgg16_bn -max_epoch=40 --resize -gpu_id=0 -maxseqlen=5 -gt_maxseqlen=1 -lr_cnn=0.0001 -lr=0.0001  -hidden_size=512 -dropout=0.5 -ngpus=2 #-num_workers=1
python3 ../src/train_vi.py -model_name=davis2016_vgg_2dataset_1clip_raw_in_recursive_fusetrain -dataset=davis2016_vi -batch_size=8 -length_clip=1 -base_model=vgg16_bn -max_epoch=40 --resize -gpu_id=0 -maxseqlen=5 -gt_maxseqlen=1 -lr_cnn=0.0001 -lr=0.0001 -hidden_size=512 -dropout=0.5 -ngpus=2 #--resume #-num_workers=1 # #-ngpus=2 -num_workers=1 #-num_workers=1 #-input_dim=6  #  --resume 
#python3 ../src/train_vi.py -model_name=davis2016_res_2dataset_1clip_opcp_raw_hpf_new -dataset=davis2016_vi -batch_size=32 -length_clip=1 -base_model=resnet34 -max_epoch=40 --resize -gpu_id=0 -maxseqlen=1 -gt_maxseqlen=1 -lr_cnn=0.0001 -lr=0.0001 -dropout=0.5 -hidden_size=512  -input_dim=9 --only_spatial #-ngpus=2 #-num_workers=1
#python3 ../src/train_refine.py -model_name=davis2016_unetaspp_flow -dataset=davis2016_vi -batch_size=8 -length_clip=2 -base_model=unet -max_epoch=40 --resize -gpu_id=0 -maxseqlen=1 -gt_maxseqlen=1 -lr_cnn=0.0001 -lr=0.001 -dropout=0.5
#python3 ../src/train_segment.py -model_name=davis2016_unetaspp_segment_3clip_bn_t3 -dataset=davis2016_vi -batch_size=4 -length_clip=3 -base_model=vgg16_bn -max_epoch=40 --resize -gpu_id=0 -maxseqlen=1 -gt_maxseqlen=1 -lr_cnn=0.0001 -lr=0.0001 -dropout=0.5 --use_segment -hidden_size=512 #-ngpus=2
