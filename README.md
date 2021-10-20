# VIDNet
Code for the [VIDNet](https://arxiv.org/pdf/2101.11080.pdf) (BMVC 2021)

## Installation

The base code is from [RVOS](https://github.com/imatge-upc/rvos).

- Install requirements ```pip install -r requirements.txt``` 

## Data

### DAVIS 2016

Download the DAVIS 2016 dataset from their [website](https://davischallenge.org/davis2016/code.html) at 480p resolution. Create a folder named ```databases```in the parent folder of the root directory of this project and put there the database in a folder named ```DAVIS2016```. The root directory (```VIDNet```folder) and the ```databases``` folder should be in the same directory.

### Inpainted DAVIS 2016
Follow the instruction on VINet to inpaint DAVIS and obtain VI inpainting result:
https://github.com/mcahny/Deep-Video-Inpainting

Follow the instruction on OPNet to inpaint DAVIS and obtain OP inpainting result:
https://github.com/seoungwugoh/opn-demo

Follow the instruction on CPNet to inpaint DAVIS and obtain CP inpainting result:
https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting

### Inpainted DAVIS 2016 ELA frame extraction
Run ```ela.py``` and change the path of ```davis_train``` and ```davis_test``` to corresponding inpainting DAVIS folder.

## Training

- Train the model for video inpainting detection with ```python train_vi.py -model_name model_name```. Checkpoints and logs will be saved under ```../models/model_name```. 

- Other arguments can be passed as well. For convenience, scripts to train with typical parameters are provided under ```scripts/```. Simply run the the following:
1. ```cd src/misc```
2. change ```__C.PATH.SEQUENCES``` and ```__C.PATH.SEQUENCES2``` in ```config.py``` to be the training data path (e.g., VI, OP, CP inpainting path)
3. ```cd ../../script```
4. update ```model_name``` for model storage
5. run ```train_davis.sh```



## Evaluation

We provide bash scripts to  evaluate models for the DAVIS 2016 and FVI datasets. You can find them under the ```scripts``` folder. 

For DAVIS evaluation:
 1. ```cd src/misc```
 2. change ```__C.PATH.SEQUENCES``` in ```config.py``` to be the testing data path (e.g., VI, OP, CP)
 3. ```cd ../../scripts```
 4. update model_name in ```eval_davis.sh```
 5. run ```eval_davis.sh```
For FVI evaluation:
 1. ```cd scripts```
 2. update model_name in ```eval_fvi.sh```
 3. run ```eval_fvi.sh```


## Citation
If this code or dataset helps your research, please cite our paper:

```
@inproceedings{zhou2021vid,
  title={Deep Video Inpainting Detection},
  author={Zhou, Peng and Yu, Ning and Wu, Zuxuan and Davis, Larry S and Shrivastava, Abhinav and Lim, Ser Nam},
  booktitle = {BMVC},
  year={2021}
}
```

