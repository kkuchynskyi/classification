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


def adjust_learning_rate(optimizer, epoch, init_lr, lr_drop_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 24))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_best(state, experiment_name):
    model_path = './training_results/' + experiment_name + '/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    torch.save(state, model_path + experiment_name + '_best.pth.tar')


def save_logs(exp_name, logs):
    print("CREATING LOG-file....")
    log_filename = os.path.join('./training_results/' + exp_name, 'history.log')
    log_file = open(log_filename,"w")
    log_file.write("epoch,train_acc,val_acc \n")
    for key,values in results_dict.items():
        log_file.write(str(key)+",")
        log_file.write(str(values[0])+",")
        log_file.write(str(values[1]))
        log_file.write("\n")
    log_file.close()
    print("LOG-file was created successfully !")


if __name__ == "__main__":
    # to be changed. e.g. 
    parameters_path = "/experiments/resnet18.yaml"
    with open(os.path.join(os.path.abspath(os.getcwd()), parameters_path)) as f:
        prms = yaml.safe_load(f)
    print("Parameters of training: ", prms)
    print("Model name = ", prms["exp_name"])

    exp_start = time.time()
    start_dataset_time = time.time()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(prms["train_dir"],
                                         transforms.Compose([transforms.RandomResizedCrop(224),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             normalize])
    )
    val_dataset = datasets.ImageFolder(prms["val_dir"],
                                       transforms.Compose([transforms.Resize(256),
                                                           transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           normalize])
    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=prms["batch_size"],
                                               shuffle=True,
                                               num_workers=prms["num_workers"],
                                               pin_memory=True,
                                               sampler=None
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
    optimizer = torch.optim.SGD(model.parameters(), lr=prms["lr"], momentum=0.9, weight_decay=1e-4)


    print("START TRAINING....")
    best_acc = 0
    results_dict = OrderedDict()
    for epoch in range(prms["init_epoch"], prms["epochs"]):
        # changing a learning rate 
        adjust_learning_rate(optimizer, epoch, prms["lr"], prms["lr_drop_step"])
        
        # train one epoch
        start_train = time.time()
        train_acc = train(train_loader, model, criterion, optimizer, epoch, print_freq=100)
        print("Elapsed time :", (time.time() - start_train)/60, " minutes!")    
        
        # validate on the validation dataset
        val_acc = validate(val_loader, model, criterion)
        
        if val_acc > best_acc:
            print("SAVING MODEL WITH updated accuracy :{}".format(val_acc))
            # save model 
            best_acc = val_acc
            save_best({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, prms["exp_name"])
        results_dict[epoch] = [train_acc, val_acc]


    exp_end = time.time()
    print("Elapsed time :", (exp_end - exp_start)/3600)
    save_logs(prms["exp_name"], results_dict)
