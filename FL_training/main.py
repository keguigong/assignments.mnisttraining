import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.generate_noniid import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from utils.update import LocalUpdate
from model import MnistNet
from FedTrainer import FedAvg
from utils.test import test_img

import logging
import os
import datetime
import json


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # NOTE: Remember to add the following to options
    # parser.add_argument('--logdir', type=str, default="./logs/", help='Log directory path')
    # init log and arg files
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    timenow = datetime.datetime.now().strftime("%m%d-%H%M%S")
    setup_name = 'fed_{}_{}_C{}_iid{}_{}'.format(args.dataset, args.epochs, args.frac, args.iid, timenow)
    argument_path = setup_name + '.json'
    log_path = setup_name + '.log'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s %(message)s') # print in the screen
    handler = logging.FileHandler(os.path.join(args.logdir, log_path)) # save to the file
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        test_dataset = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(train_dataset, args.num_users)
        else:
            dict_users = mnist_noniid(train_dataset, args.num_users)
    img_size = train_dataset[0][0].shape

    # build model
    net_glob = MnistNet().to(args.device)
    print(net_glob)
    net_glob.train()

    # copy weights(global model)
    w_glob = net_glob.state_dict()

    # training
    total_train_loss = []
    total_train_acc = []
    total_test_loss = []
    total_test_acc = []
        
    for n_iter in range(args.epochs):
        loss_locals = []

        local_test_loss = []
        local_test_acc = []

        w_locals = []
        m = max(int(args.frac * args.num_users), 1)     # client selection, default = 0.1
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users[idx], test_dataset=test_dataset)   # local update
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))   # local training

            local_net = MnistNet().to(args.device)
            local_net.load_state_dict(w)
            test_loss, test_acc = test_img(local_net, test_dataset, args)

            w_locals.append(copy.deepcopy(w))
            
            loss_locals.append(copy.deepcopy(loss))
            local_test_loss.append(test_loss)
            local_test_acc.append(test_acc)
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # test and print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        test_loss_avg = sum(local_test_loss) / len(local_test_loss)
        test_acc_avg = sum(local_test_acc) / len(local_test_acc)
        print('Round {:3d} || Average train loss {:.3f} || Average test loss {:.3f} || Average test accuracy {:.3f}'.format(n_iter, loss_avg, test_loss_avg, test_acc_avg))
        logger.info('Round {:3d} || Average train loss {:.3f} || Average test loss {:.3f} || Average test accuracy {:.3f}'.format(n_iter, loss_avg, test_loss_avg, test_acc_avg))       
        total_train_loss.append(loss_avg)
        total_test_loss.append(test_loss_avg)
        total_test_acc.append(test_acc_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(total_train_loss)), total_train_loss)
    plt.ylabel('training loss')
    plt.savefig('./save/fed_{}_{}_C{}_iid{}.png'.format(args.dataset, args.epochs, args.frac, args.iid))

    # plt.figure()
    # plt.plot(range(len(total_test_acc)), total_test_acc)
    # plt.ylabel('testing accuracy')
    # plt.savefig('./save/fed_{}_{}_C{}_iid{}.png'.format(args.dataset, args.epochs, args.frac, args.iid))

    # testing
    # net_glob.eval()
    # # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))
