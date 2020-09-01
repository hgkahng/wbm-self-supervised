# Self-Supervised Learning for Wafer Bin Map Classification

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [PyTorch](https://pytorch.org)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [kornia](https://github.com/kornia/kornia)
```
conda update -n base conda  # use 4.8.4 or higher
conda create -n wbm python=3.6
conda activate wbm
conda install anaconda
conda install opencv -c conda-forge
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install pytorch_lightning
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
    --warmup_steps 5 \
    --checkpoint_root "./checkpoints/" \
    scratch
```
- Run ```python run_classification.py --help``` for more optional arguments.
- If running on a Windows machine, set `num_workers` to 0. (multiprocessing does not function well.)
