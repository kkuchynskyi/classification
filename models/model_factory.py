import torch

from models.vgg16_bn import VGG16BN
from models.resnet import ResNet18, ResNet50
#from models.resnet50 import ResNet50
from models.mobilenet_v2 import MobileNetV2


def create_model(model_prms):
    if model_prms["type"] == "vgg16_bn":
        model = VGG16BN(model_prms)
    elif model_prms["type"] == "resnet18":
        model = ResNet18()
    elif model_prms["type"] == "resnet50":
        model = ResNet50()#.cuda()
    elif model_prms["type"] == "mobilenet_v2":
        model = MobileNetV2(model_prms)
    else:
        raise KeyError("Model type not in ['vgg16_bn', 'resnet18', 'mobilenet_v2', 'resnet50']")


    print("{} model was created succesfully!".format(model_prms["type"]))
    if model_prms["pretrained_model"] is not None:
        model.load_state_dict(torch.load(model_prms["pretrained_model"])["state_dict"])
        print("Weights was fully downloaded from {}!".format(model_prms["pretrained_model"]))

    return model