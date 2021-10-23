import os 
import time
import yaml

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import validate, speed_evaluation
from models.model_factory import create_model


if __name__ == "__main__":
    # to be changed. e.g. 
    parameters_path = "./configs/resnet50.yaml"
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
    model = create_model(model_prms)

    criterion = nn.CrossEntropyLoss().cuda()

    # validate on the validation dataset
    val_acc = validate(val_loader, model, criterion)
    
    speed_evaluation(model)