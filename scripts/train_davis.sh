#!/bin/bash

python3 ../src/train_vi.py -model_name=davis2016_vgg_2dataset_3clip_raw_in_recursive_fusetrain -dataset=davis2016_vi -batch_size=3 -length_clip=3 -base_model=vgg16_bn -max_epoch=40 --resize -gpu_id=0 -maxseqlen=5 -gt_maxseqlen=1 -lr_cnn=0.0001 -lr=0.0001 -hidden_size=512 -dropout=0.5 #-ngpus=2 #--resume #-num_workers=1 #-input_dim=6  #  --resume 

