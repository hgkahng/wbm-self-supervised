# Self-Supervised Learning for Wafer Bin Map Classification


## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [entmax](https://github.com/deep-spin/entmax)
- [kornia](https://github.com/kornia/kornia)
```
conda create -n wbm python=3.6
conda activate wbm
conda install anaconda
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install opencv -c opencv
pip install pytorch_lightning
pip install entmax
pip install --no-deps kornia  # to be removed in future distributions
```

## Dataset
1. Download from the following link: [WM-811K](https://www.kaggle.com/qingyi/wm811k-wafer-map)
2. Place the `LSWMD.pkl` file under `./data/wm811k/`.
3. Run the following script from the working directory:
```
python process_wm811k.py
```

## Usage
### Train classification model from scratch (no self-supervised pre-training)
```
python run_classification.py \
    --data wm811k \
    --input_size 96 \
    --backbone_type alexnet \
    --backbone_config batch_norm \
    --balance \
    --optimizer "sgd" \
    --learning_rate 1e-2 \
    --weight_decay 1e-3 \
    --momentum 9e-1 \
    --scheduler "cosine" \
    --warmup_steps 5 \
    --checkpoint_root "./checkpoints/" \
    scratch
```
Run ```python run_classification.py --help``` for more optional arguments.
