# Imagenet training
This repository contains a pipeleine for training on Imagenet dataset. It is designed for my personal use, but can be helpful for everyone who with to train models on its own. All models trained and evaluated on RTX 3060 + PyTorch.  


## Installation
```
pip install -r requirments.txt
```



## Evaluation model
Download model from table. Choose config file(e.g. `configs/resnet18.yaml`), inside of it specify `val_dir`, `pretrained_model` for your local machine parameters. In `test.py` chenge path to config file:
```
    parameters_path = "/configs/resnet18.yaml"
```
Then run evaluation:
```
python test.py
```
## Train model

## Imagenet Dataset
You can download [zip archive](https://drive.google.com/drive/folders/1-C0K1vT4YQe4cZgZ_XV_U_IMoCH3wwbt?usp=sharing), total zip archive devided by 10Gb archives.
Merge them to one:
```
zip -a imagente
```
And extract atchive
```
pass
```
