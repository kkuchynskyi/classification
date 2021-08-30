import os 
import time
import yaml
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import validate, train
from models.vgg16_bn import VGG16BN
from models.resnet18 import ResNet18
from models.mobilenet_v2 import MobileNetV2


if __name__ == "__main__":
    # to be changed. e.g. 
    parameters_path = "/configs/resnet18.yaml"
    with open(os.path.join(os.path.abspath(os.getcwd()), parameters_path)) as f:
        prms = yaml.safe_load(f)
    print("Parameters of training: ", prms)
    print("Model name = ", prms["exp_name"])

    exp_start = time.time()
    start_dataset_time = time.time()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(prms["val_dir"],
                                       transforms.Compose([transforms.Resize(256),
                                                           transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           normalize])
    )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=prms["batch_size"],
                                             shuffle=False,
                                             num_workers=prms["num_workers"],
                                             pin_memory=True
    )
    print('Elapsed time on the data loaders=', time.time() - start_dataset_time)

    model_prms = prms["model"]
    if model_prms["type"] == "vgg16_bn":
        model = VGG16BN(model_prms).cuda()
    elif model_prms["type"] == "resnet18":
        model = ResNet18(model_prms).cuda()
    elif model_prms["type"] == "mobilenet_v2":
        model = MobileNetV2(model_prms).cuda()
    else:
        raise KeyError("Model type not in ['vgg16_bn', 'resnet18', 'mobilenet_v2']")

    if model_prms["pretrained_model"] is not None:
        model.load_state_dict(torch.load(model_prms["pretrained_model"])["state_dict"])

    
    criterion = nn.CrossEntropyLoss().cuda()


    # validate on the validation dataset
    val_acc = validate(val_loader, model, criterion)
