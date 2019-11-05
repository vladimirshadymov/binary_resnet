from __future__ import print_function
from vgg import vgg16, vgg19
from training_routines import train, test
import torch.nn as nn
from training_routines import train, test
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 BNN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda-num', type=int, default=0,
                        help='Choses GPU number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--weight-decay', type=float, default=0, metavar='W',
                        help='coefficient of L2 regulariztion')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model', type=str, default='vgg16',
                        help='Model choice')
    parser.add_argument('--binarized', action='store_true', default=False,
                        help='Makes model binary')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:%d" % args.cuda_num if torch.cuda.is_available() else "cpu")
    print("Use device:", device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageNet('../data/', split='train', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    train_loader_aug = torch.utils.data.DataLoader(
        datasets.ImageNet('../data/', split='train', download=True,
                       transform=transforms.Compose([
                           transforms.RandomAffine(degrees=35, shear=0.2),
                           transforms.RandomCrop(224, padding=5),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageNet('../data/', split='val', download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    test_loader_aug = torch.utils.data.DataLoader(
        datasets.ImageNet('../data/', split='val', download=True,
        transform=transforms.Compose([
            transforms.RandomAffine(degrees=35, shear=0.2),
            transforms.RandomCrop(224, padding=5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.model=='vgg16':
        model = vgg16(binarized=args.binarized).to(device)
    elif args.model=='vgg19':
        model = vgg19(binarized=args.binarized).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # managinng lr decay

    test_accuracy = []
    train_accuracy = []

    for epoch in range(1, args.epochs + 1):
        print('Epoch:', epoch, 'LR:', scheduler.get_lr())
        train(args, model, device, train_loader, optimizer, epoch)
        
        print("Train set:\n")
        test(args, model, device, train_loader, optimizer, epoch)
        print("Test set:\n")
        test(args, model, device, test_loader, optimizer, epoch)

        print("Train augmented set:\n")
        test(args, model, device, train_loader_aug, optimizer, epoch)
        print("Test augmented set:\n")
        test(args, model, device, test_loader_aug, optimizer, epoch)
        scheduler.step(epoch=epoch)
        # if epoch > 10:
        #     if (args.save_model):
        #         torch.save(model.state_dict(), "../model/cifar10_conv_bnn.pt")

if __name__ == '__main__':
    main()
