export CUDA_VISIBLE_DEVICES=0
export PATH="/vulcan/scratch/pengzhou/tools/anaconda3/bin:${PATH}"
export PYTHONPATH="/vulcan/scratch/pengzhou/tools/anaconda3/lib/python3.6/site-packages:${PYTHONPATH}"
export LD_LIBRARY_PATH="/vulcan/scratch/pengzhou/tools/anaconda3/lib/:${LD_LIBRARY_PATH}"
#python3 ../src/eval.py -model_name=spatiotemporal_davis_bs_04_lc_05_256p_from_youtube -dataset=davis2016 -eval_split=val -batch_size=1 -length_clip=130 -gpu_id=0
#python3 ../src/eval.py -model_name=davis2016_vgg_2dataset_1clip_raw_in_recursive_fusetrain -dataset=davis2016 -eval_split=val -batch_size=1 -length_clip=130 -gpu_id=0 --overlay_masks -maxseqlen=5 -hidden_size=512 -num_workers=1 #-input_dim=9
#python3 ../src/eval_edge.py -model_name=davis2016_vgg_2dataset_3clip_raw_in_edge_recursive -dataset=davis2016 -eval_split=val -batch_size=1 -length_clip=130 -gpu_id=0 --overlay_masks -maxseqlen=5 -hidden_size=512
python3 ../src/eval_ela.py -model_name=davis2016_vgg_2dataset_3clip_raw_in_recursive_fusetrain -dataset=davis2016 -eval_split=val -batch_size=1 -length_clip=130 -gpu_id=0 --overlay_masks -maxseqlen=5 -hidden_size=512 -num_workers=1 #-input_dim=9
#python3 ../src/eval_previous_mask.py -model_name=davis2016_vi_completion_2_prev_mask -dataset=davis2016 -eval_split=val -batch_size=1 -length_clip=130 -gpu_id=0 --overlay_masks