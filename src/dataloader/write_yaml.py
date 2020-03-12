import yaml
import os
from collections import defaultdict
import pdb
import glob
dicts = [{'attributes':['AC', 'BC', 'CS']},
{'sets':  ['train', 'val', 'val-dev']},
{'years': [2016, 2017]},
{'sequences':[]}]

#category={'name':None,'attributes':[],'num_frames':0,'set':None,'eval_t':True,'year':2016}



dirs = glob.glob('/vulcan/scratch/pengzhou/model/Free-Form-Video-Inpainting/test_outputs/epoch_0/test_object_like/*/')

for d in dirs:
	category= {}
	category['name']=d.split('/')[-2]
	category['attributes']=[]
	category['num_frames']=15
	category['set']='val'
	category['eval_t']=True
	category['year']=2016
	#pdb.set_trace()
	dicts[-1]['sequences'].append(category)
with open('freeform_db.yaml', 'w') as yaml_file:
	yaml.dump(dicts[0], yaml_file, default_flow_style=False)
	yaml.dump(dicts[1], yaml_file, default_flow_style=False)
	yaml.dump(dicts[2], yaml_file, default_flow_style=False)
	yaml.dump(dicts[3], yaml_file, default_flow_style=False)
