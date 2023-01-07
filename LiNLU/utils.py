from __future__ import print_function
import torch
import numpy as np
import torchvision
from torchvision import transforms,datasets

from networks_and_functions import MLP
from networks_and_functions import AlexNet
from networks_and_functions import VGG16
from networks_and_functions import ResNet18


def load_data(task, batch_size):
    if task == 'CIFAR-10':        
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             normalize])
        
        data_path = './Dataset'
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)    
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    
    elif task == 'ImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transformation_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(), 
                                                   normalize])
        
        transformation_val = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(), 
                                                 normalize])
        
        data_path = './Dataset/ImageNet'
        train_set = datasets.ImageFolder(data_path+'/train', transformation_train)
        val_set = datasets.ImageFolder(data_path+'/val', transformation_val)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        
    
    return train_loader, test_loader



def make_model(network, task, activation):
    if network == 'MLP':
        return MLP(task, activation)
    elif network == 'AlexNet':
        return AlexNet(task, activation)
    elif network == 'VGG16':
        return VGG16(task, activation)
    elif network == 'ResNet18':
        return ResNet18(task, activation)
    else:
        print("=== Enter the correct network name.")


def save_model(names, model, optimizer, acc, epoch, acc_hist, train_loss_hist, test_loss_hist, alpha_hist=None):
    state = {
        'net': model.state_dict(),
        'opt': optimizer.state_dict(),
        'acc': acc,
        'acc_hist': acc_hist,
        'epoch': epoch,
        'loss_train_hist': train_loss_hist,
        'loss_test_hist': test_loss_hist,
        'alpha_hist' : alpha_hist
    }
    
    torch.save(state, './' + '{}.pth'.format(names))
    
    # best test accuracy
    best_acc = max(acc_hist)
    
    if acc == best_acc:
        torch.save(state, './' + '{}_best.pth'.format(names))


# Optimizer scheduler
def scheduler_step(opt, epoch, decay_interval, gamma) : 
    lr = []
    if epoch % decay_interval == 0 :
        for g in opt.param_groups:
            g['lr'] = g['lr']*gamma
            lr.append(g['lr'])
    
        print("=== learning rate decayed to {}.\n".format(lr))

