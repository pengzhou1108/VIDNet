export CUDA_VISIBLE_DEVICES=0
export PATH="/vulcan/scratch/pengzhou/tools/anaconda3/bin:${PATH}"
export PYTHONPATH="/vulcan/scratch/pengzhou/tools/anaconda3/lib/python3.6/site-packages:${PYTHONPATH}"
export LD_LIBRARY_PATH="/vulcan/scratch/pengzhou/tools/anaconda3/lib/:${LD_LIBRARY_PATH}"
#python3 ../src/eval.py -model_name=spatiotemporal_davis_bs_04_lc_05_256p_from_youtube -dataset=davis2016 -eval_split=val -batch_size=1 -length_clip=130 -gpu_id=0
#python3 ../src/iou_measure.py --input='../models/davis2016_vgg_2dataset_1clip_raw_in_recursive_fusetrain/masks_model' --dataset=davis2016 --mask_dir='/vulcan/scratch/pengzhou/dataset/DAVIS/Annotations/480p' --im_dir='/vulcan/scratch/pengzhou/model/Copy-and-Paste-Networks-for-Deep-Video-Inpainting/val'
#python3 ../src/iou_measure.py --input='../models/davis2016_vgg_2dataset_1clip_raw_in_recursive_fusetrain/masks_results' --dataset=davis2016 --mask_dir='/vulcan/scratch/pengzhou/dataset/DAVIS/Annotations/480p' --im_dir='/vulcan/scratch/pengzhou/model/Deep-Video-Inpainting/results/vi_davis/val'
python3 ../src/iou_measure.py --input='../models/davis2016_vgg_2dataset_1clip_raw_in_recursive_fusetrain/masks_opn-demo' --dataset=davis2016 --mask_dir='/vulcan/scratch/pengzhou/dataset/DAVIS/Annotations/480p' --im_dir='/vulcan/scratch/pengzhou/model/opn-demo/vi_davis/val'
#python3 ../src/iou_measure.py --input='../models/davis2016_vgg_2dataset_3clip_raw_in_recursive_new/masks_test_outputs' --dataset=davis2016 --mask_dir='/vulcan/scratch/pengzhou/model/Free-Form-Video-Inpainting/FVI/Test/object_masks' --im_dir='/vulcan/scratch/pengzhou/model/Free-Form-Video-Inpainting/test_outputs/epoch_0/test_object_like'
