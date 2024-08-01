import os
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import LeNet
from kmm import kmm, get_kernel_width
from dataloader import load_MNIST


parser = argparse.ArgumentParser()
parser.add_argument("--noise_type", type = str, default = 'symmetric')
parser.add_argument("--noise_rate", type = float, default = 0.4)

parser.add_argument("--num_val", type = int, default = 1000)
parser.add_argument("--batch_size", type = int, default = 256)
parser.add_argument("--batch_size_val", type = int, default = 256)
parser.add_argument("--num_epoch", type = int, default = 400)

parser.add_argument("--lr", type = float, default = 0.0003)
parser.add_argument("--step", type = float, default = 100)
parser.add_argument("--gamma", type = float, default = 0.1)
parser.add_argument("--weight_decay", type = float, default = 0.0001)

parser.add_argument("--seed", type = int, default = 100)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.environ['PYTHONHASHSEED'] = '0'
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def get_model():
    net = LeNet(n_out = 10)

    if torch.cuda.is_available():
        net.cuda()

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step, gamma=args.gamma)

    return net, opt, scheduler


def main():

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # data loaders
    train_loader, val_loader, test_loader = load_MNIST(
        args.batch_size, 
        args.batch_size_val,
        args.noise_type,
        args.noise_rate, 
        args.num_val)

    # define the model, optimizer, and lr decay scheduler
    net, opt, scheduler = get_model()

    # train the model
    test_acc = []

    for epoch in range(args.num_epoch):

        train_acc_tmp = []
        test_acc_tmp = []

        with tqdm(total = len(train_loader), desc = f"Epoch count: {epoch} / {args.num_epoch}", unit="batch") as pbar: 
            for train_data, train_labels, _ in train_loader:

                # estimate the w(x) by the loss distribution between train and test. 
                net.eval()

                train_data = train_data.to(device)
                train_labels = train_labels.to(device)

                result_train = net(train_data)
                l_tr = F.cross_entropy(result_train, train_labels, reduction='none') # calculate the loss of train data. 

                val_data, val_labels, _ = next(iter(val_loader)) # manually loading next val mini-batch. 

                val_data = val_data.to(device)
                val_labels = val_labels.to(device)

                result_val = net(val_data) 
                l_val = F.cross_entropy(result_val, val_labels, reduction='none') # calculate the loss of test data. 


                l_tr_reshape = np.array(l_tr.detach().cpu()).reshape(-1, 1) # detach them from gradient calculation. 
                l_val_reshape = np.array(l_val.detach().cpu()).reshape(-1, 1)

                # warm start
                if epoch < 1:
                    # In the 1st epoch it is warm up. 
                    # Coefficient vector is fixed to 1. 
                    coef = [1 for i in range(len(_))]
                else:
                    # Using Kernel Mean Matching to calculate the coefficient vector. 
                    kernel_width = get_kernel_width(l_tr_reshape)
                    # KMM's dimension is the dimension of data in train domain. 
                    coef = kmm(l_tr_reshape, l_val_reshape, kernel_width) 

                # Using this w to construct w(x). 
                w = torch.from_numpy(np.asarray(coef)).float().to(device)

                # Train the classifier by train data. 
                # Using w(x) we estimated. 
                net.train()
                result_train_wc = net(train_data)
                l_tr_wc = F.cross_entropy(result_train_wc, train_labels, reduction='none') 
                l_tr_wc_weighted = torch.sum(l_tr_wc * w) # multiply the w(x)

                # Learning step. 
                opt.zero_grad()
                l_tr_wc_weighted.backward()
                opt.step()

                # memory the accuracy for minibatch train data. 
                train_correct = 0
                train_total = 0
                _, predicted = torch.max(result_train_wc.data, 1)

                train_total += train_labels.size(0)
                train_correct += (predicted == train_labels).sum().item()
                train_accuracy = train_correct / train_total
                train_acc_tmp.append(train_accuracy)

                pbar.update(1)

        train_accuracy_mean = np.mean(train_acc_tmp)
        print(f"average of test accuracy in this epoch is {train_accuracy_mean}")

        net.eval()
        # evaluate the accuracy in test domain. 
        for test_img, test_label, _ in test_loader:
            test_img = test_img.to(device)
            test_label = test_label.to(device)

            test_correct = 0
            test_total = 0

            result_test = net(test_img)
            _, predicted = torch.max(result_test.data, 1)

            test_total += test_label.size(0)
            test_correct += (predicted == test_label).sum().item()
            test_accuracy = test_correct / test_total

            test_acc_tmp.append(test_accuracy)

        test_accuracy_mean = np.mean(test_acc_tmp)
        print(f"average of train accuracy in this epoch is {test_accuracy_mean}")
        test_acc.append(test_accuracy_mean)
        test_acc_arr = np.array(test_acc)

        scheduler.step()

    # save the output
    np.savetxt('./output/test_acc.txt', test_acc_arr, fmt='%s')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(test_acc)
    fig.savefig('./output/test_acc.png')


if __name__ == '__main__':
    main()
