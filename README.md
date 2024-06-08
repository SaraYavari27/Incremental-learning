# CL_Medical_Images

Note: Please change “log_dir” based on your own data directory.  


## Usage

Train and test:
```
python train_test.py \
    --lr 1e-5 \
    --lambda_cov 0.8 \
    --first_bn_mul 10.0 \
    --BatchSize 64 \
    --epochs 50 \
    --num_instances 2 \
    --batch_size_gen 16 \
    --num_instances_gen 8 \
    --data 'PICAI' \
    --loss 'triplet' \
    --log_dir 'PICAI' \
    --epoch_gen 100 \
    --momentum 0.9 \
    --weight-decay 2e-4 \
    --momentum_0 0.9 \
    --weight-decay_0 5e-4 \
    --epochs_0 50 \
    --lr_0 0.001 \
    --BatchSize_0 64 \
    --task 4 \
    --base 2 \
    

    

---------

The directory tree structure for the "Dataset" folder should be like this (containing images):


├── PICAI
│   ├── train
│   │  │──case-ISUP0
│   │  ├──case-ISUP1
│   │  ├──case-ISUP2
│   │  └──case-ISUP3
│   │  
│   │── test
│   │  │──case-ISUP0
│   │  ├──case-ISUP1
│   │  ├──case-ISUP2
│   │  └──case-ISUP3


## Requirements
- python 3.9+
- torch 1.10.1+
- torchvision 0.11.2+
- numpy 1.22.0+
- SimpleITK 2.2.1+
