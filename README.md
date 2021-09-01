# Imagenet training
This repository contains a pipeleine for training on Imagenet dataset. It is designed for my personal use, but can be helpful for everyone who wish to train models on its own. All models trained and evaluated on RTX 3060 + PyTorch.  

| Model | Accuracy@1 | Accuracy@5 | FPS | Log | Weights |
| --- | --- | --- | --- | --- | --- |
| VGG16 with bn | 73.036 | 91.270 | 124.53 | [log](https://drive.google.com/file/d/1ejryvdIAeQXf6mRLCSdxaaalHc1dYvUH/view?usp=sharing)| [weight](https://drive.google.com/file/d/16H-NDYwCB7AimxIdIDRkF6QuLm8MQa0j/view?usp=sharing)| 
| ResNet18 | 65.954| 86.730 | 211.93 | [log](https://drive.google.com/file/d/1gGvlGhNw18VT_i_dFoqJG_sIH3K5AAmt/view?usp=sharing) | [weight](https://drive.google.com/file/d/128_895Oe9gxiBfLaYPJFnOgthcF4DHP5/view?usp=sharing) |
| MobileNetV2 alpha 1 | 69.330 | 88.854 | 222.26 | [log](https://drive.google.com/file/d/16OuLP3MW7i3O17FlsHDkxo0ByAODjrgy/view?usp=sharing) | [weight](https://drive.google.com/file/d/1gUoJko-RpxoxXKZ72dt4e20kUJHGqObJ/view?usp=sharing) |


## Installation
```
git clone https://github.com/kkuchynskyi/classification.git
pip install -r requirments.txt
```



## Evaluation model
Download model from table. Choose config file(e.g. `configs/resnet18.yaml`), inside of it specify `val_dir`, `pretrained_model` for your local machine parameters. In `test.py` 18 line change path to config file:
```
parameters_path = "./configs/resnet18.yaml"
```
Then run evaluation:
```
python test.py
```
## Train model
In `train.py` change 47 line to the model config file e.g. `resnet18.yaml`, `mobilenet_v2.yaml` and `vgg16_bn.yaml`
```
parameters_path = "./configs/resnet18.yaml"
```


## Imagenet Dataset
You can download [zip archive](https://drive.google.com/drive/folders/1-C0K1vT4YQe4cZgZ_XV_U_IMoCH3wwbt?usp=sharing), total zip archive devided by 10Gb archives.
Merge them to one:
```
cat imagenet* > imagenet2012.tar.gz
```
And extract atchive
