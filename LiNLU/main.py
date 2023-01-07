from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from utils import load_data
from utils import make_model
from utils import save_model
from utils import scheduler_step

from networks_and_functions import param_init
from networks_and_functions import param_load
from networks_and_functions import p_load


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser(description='LiNLNet: Gauging Required Nonlinearity in Deep Neural Networks')

parser.add_argument('--task', type=str, default='CIFAR-10', help='which task to run (CIFAR-10, ImageNet)')
parser.add_argument('--network', type=str, default='MLP', help='which network to run (MLP, AlexNet, VGG16, ResNet18)')
parser.add_argument('--mode', type=str, default='train', help='whether to train or eval')
parser.add_argument('--activation', type=str, default='LiNLU', help='which activation function to use(ReLU, LiNLU)')

# Hyperparameters 
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='Maximum number of epochs')
parser.add_argument('--weight_decay', type=float, default=5E-4, help='Weight decay (L2 regularization) coefficient')
parser.add_argument('--learning_rate', type=float, default=1E-2, help='Model learning rate')
parser.add_argument('--linlu_learning_rate', type=float, default=1E-3, help='LiNLU learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
parser.add_argument('--lr_decay_interval', type=int, default=140, help='Learning rate decay interval')


args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    names = args.network + '_' + args.task + '_' + args.activation
    train_loader, test_loader = load_data(args.task, args.batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    
    if args.mode == 'train':
        model = make_model(args.network, args.task, args.activation).to(device)
        param_init(model, args.task)
        
        acc_hist = list([])
        train_loss_hist = list([])
        test_loss_hist = list([])
        
        if args.activation == 'ReLU':
            p_hist = None
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
        
        elif args.activation == 'LiNLU':
            p_hist = list([])
            model_params, linlu_params = param_load(model, args.network)
            optimizer = torch.optim.SGD(
                [{'params' : model_params, 'lr' : args.learning_rate, 'momentum' : 0.9, 'weight_decay' : args.weight_decay},
                 {'params' : linlu_params, 'lr' : args.linlu_learning_rate, 'momentum' : 0.9, 'weight_decay' : args.weight_decay}])
        
        
        for epoch in range(args.num_epochs):
            if epoch != 0 :
                scheduler_step(optimizer, epoch, args.lr_decay_interval, args.lr_decay_rate)
            
            start_time = time.time()
            train_loss = train(model, train_loader, criterion, optimizer)
            test_loss, acc = test(model, test_loader, criterion)
            
            acc_hist.append(acc)
            train_loss_hist.append(train_loss)
            test_loss_hist.append(test_loss)
            time_elapsed = time.time() - start_time
            
            print("\nEpoch: {}/{}.. ".format(epoch+1, args.num_epochs).ljust(14),
                      "Train Loss: {:.3f}.. ".format(train_loss).ljust(20),
                      "Test Loss: {:.3f}.. ".format(test_loss).ljust(19),
                      "Test Accuracy: {:.3f}".format(acc))        
            print("Time elapsed: {:.6f}".format(time_elapsed))
            
            if args.activation == 'LiNLU': 
                p_hist.append(p_load(linlu_params))
                print("p values: {}".format(p_hist[-1]))
            
            # save model pth file
            save_model(names, model, optimizer, acc, epoch, acc_hist, train_loss_hist, test_loss_hist, p_hist)
            
            
    elif args.mode == 'eval':
        model = make_model(args.network, args.task, args.activation).to(device)
        c = torch.load("./{}_best.pth".format(names))
        model.load_state_dict(c['net'])
        
        test_loss, acc = test(model, test_loader, criterion)
        print("Test Accuracy: {:.3f}".format(acc))
        
        
def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        output = model(images.to(device))
        
        loss = F.cross_entropy(output, labels.to(device))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() / len(train_loader)
        
    return train_loss


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    len_dataset = len(test_loader.dataset)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            output = model(inputs.to(device))
            
            loss = criterion(output.cpu(), targets)
            test_loss += loss.item() / len(test_loader)
            
            output = F.softmax(output, dim=1)
            pred = torch.argmax(output.cpu(), dim=1)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            
            acc = 100. * correct / len_dataset
            
    return test_loss, acc


if __name__=='__main__':
    main()
