# VIDNet

## Installation

- Install requirements ```pip install -r requirements.txt``` 

## Data

### DAVIS 2016

Download the DAVIS 2016 dataset from their [website](https://davischallenge.org/davis2016/code.html) at 480p resolution. Create a folder named ```databases```in the parent folder of the root directory of this project and put there the database in a folder named ```DAVIS2016```. The root directory (```VIDNet```folder) and the ```databases``` folder should be in the same directory.

### Inpainted DAVIS 2016
Follow the instruction on VINet to inpaint DAVIS:
https://github.com/mcahny/Deep-Video-Inpainting

Follow the instruction on OPNet to inpaint DAVIS:
https://github.com/seoungwugoh/opn-demo

Follow the instruction on CPNet to inpaint DAVIS:
https://github.com/shleecs/Copy-and-Paste-Networks-for-Deep-Video-Inpainting

### DAVIS 2016 ELA frame extraction
Run ```ela.py``` and change the path of ```davis_train``` and ```davis_test``` to corresponding inpainting DAVIS folder.

## Training

- Train the model for video inpainting detection with ```python train_vi.py -model_name model_name```. Checkpoints and logs will be saved under ```../models/model_name```. 

- Other arguments can be passed as well. For convenience, scripts to train with typical parameters are provided under ```scripts/```. Simply run the the following:

```cd script```
run ```train_davis.sh```


## Evaluation

We provide bash scripts to  evaluate models for the DAVIS 2017 and FVI datasets. You can find them under the ```scripts``` folder. On the one hand, ```eval_davis.sh```and ```eval_fvi.sh``` respectively. 


