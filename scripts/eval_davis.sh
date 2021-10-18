
python3 ../src/eval.py -model_name=davis2016_vgg_2dataset_3clip_raw_in_recursive_fusetrain -dataset=davis2016 -eval_split=val -batch_size=1 -length_clip=130 -gpu_id=0 --overlay_masks -maxseqlen=5 -hidden_size=512 -num_workers=1
#python3 ../src/eval_ela.py -model_name=davis2016_vgg_2dataset_3clip_raw_in_recursive_fusetrain -dataset=davis2016 -eval_split=val -batch_size=1 -length_clip=130 -gpu_id=0 --overlay_masks -maxseqlen=5 -hidden_size=512 -num_workers=1 #-input_dim=9
