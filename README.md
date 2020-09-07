# Self-Supervised Learning for Wafer Bin Map Classification

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [PyTorch](https://pytorch.org) (tested on 1.6.0)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) (tested on 0.8.5)
- [Albumentations](https://github.com/albumentations-team/albumentations) (tested on 0.4.6)
```
conda update -n base conda  # use 4.8.4 or higher
conda create -n wbm python=3.6
conda activate wbm
conda install anaconda
conda install opencv -c conda-forge
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install pytorch_lightning
pip install albumentations
```

## Dataset
1. Download from the following link: [WM-811K](https://www.kaggle.com/qingyi/wm811k-wafer-map)
2. Place the `LSWMD.pkl` file under `./data/wm811k/`.
3. Run the following script from the working directory:
```
python process_wm811k.py
```

## Classification (from scratch)
### 1. From the command line
```
python run_classification.py \
    --data wm811k \
    --input_size 96 \
    --backbone_type "alexnet" \
    --backbone_config "batch_norm" \
    --balance \
    --device "cuda:0" \
    --num_workers 4 \
    --optimizer "sgd" \
    --learning_rate 1e-2 \
    --weight_decay 1e-3 \
    --momentum 9e-1 \
    --scheduler "cosine" \
    --warmup_steps 0 \
    --checkpoint_root "./checkpoints/" \
    scratch
```
- Run ```python run_classification.py --help``` for more optional arguments.
- If running on a Windows machine, set `num_workers` to 0. (multiprocessing does not function well.)

### 2. Using bash scripts
```
./experiments/classification.sh
```


## Mixup (from scratch)
### 1. From the command line
```
python run_mixup.py \
    --data wm811k \
    --input_size 96 \
    --backbone_type "alexnet" \
    --backbone_config "batch_norm" \
    --balance \
    --device "cuda:0" \
    --num_workers 4 \
    --optimizer "sgd" \
    --learning_rate 1e-2 \
    --weight_decay 1e-3 \
    --momentum 9e-1 \
    --scheduler "cosine" \
    --warmup_steps 0 \
    --checkpoint_root "./checkpoints/" \
```
- Adding a `--disable_mixup` flag will resort to simple classification withou mixup. This is implemented for the ease of making comparisons only.
### 2. Using bash scripts
```
./experiments/mixup.sh
```