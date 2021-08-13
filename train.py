import os 
import shutil
import time
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image

from utils import AverageMeter, accuracy, validate,train,validate,parse_valtxt, ValidationFromDataFrame
from models.vgg16_bn import VGG16BN

### PARAMETERS TO CHANGE 
traindir = '/media/datasets/imagenet_2012/images/train/'
valdir = '/media/datasets/imagenet_2012/images/val/'

BATCH_SIZE = 50
LEARNING_RATE = 0.01
EPOCHS = 75
INIT_EPOCH = 16
EXPERIMENT_NAME = 'vgg16_bn_from_16'
WORKERS = 20
MODEL_TYPE = 'vgg16_bn'
model_prms = {"alpha": 1}
###
print("Model name = ",EXPERIMENT_NAME)

exp_start = time.time()
start_time = time.time()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(traindir,
                                     transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])
)
val_dataset = datasets.ImageFolder(valdir,
                                   transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize])
)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=WORKERS,
                                           pin_memory=True,
                                           sampler=None
)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=WORKERS,
                                         pin_memory=True
)
print('Elapsed time on the data loaders=', time.time() - start_time)

if MODEL_TYPE == 'vgg16_bn':
    model = VGG16BN(model_prms).cuda()
    model.load_state_dict(torch.load("/home/kirill/workspace/classifiaction/training_results/vgg16_bn_from_11/vgg16_bn_from_11_best.pth.tar")["state_dict"])


criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 24))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_best(state, experiment_name):
    model_path = './training_results/' + experiment_name + '/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    torch.save(state, model_path + experiment_name + '_best.pth.tar')


print("START TRAINING....")
best_acc = 0
results_dict = OrderedDict()
for epoch in range(INIT_EPOCH, EPOCHS):
    # changing a learning rate 
    adjust_learning_rate(optimizer, epoch, LEARNING_RATE)
    
    # train one epoch
    start_train = time.time()
    train_acc = train(train_loader, model, criterion, optimizer, epoch, print_freq=100)
    end_train = time.time()
    print("Elapsed time :", (end_train - start_train)/60, " minutes!")    
    
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
            }, EXPERIMENT_NAME)
    results_dict[epoch] = [train_acc, val_acc]


exp_end = time.time()
print("Elapsed time :", (exp_end - exp_start)/3600)    
print("CREATING LOG-file....")
log_filename = './training_results/' + EXPERIMENT_NAME + '/' + EXPERIMENT_NAME+ '.log'
log_file = open(log_filename,"w")
log_file.write("epoch,train_acc,val_acc \n")
for key,values in results_dict.items():
    log_file.write(str(key)+",")
    log_file.write(str(values[0])+",")
    log_file.write(str(values[1]))
    log_file.write("\n")
log_file.close()
