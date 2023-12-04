#from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time

class InferenceLatencyUtil:
    def __init__(self):
        self.savedTime = time.time()
        self.enabled = False

    def time_flag1(self):
        self.savedTime = time.time()
        return self.savedTime   
    
    def time_flag2(self, text):
        now = time.time()
        deltaTime = now - self.savedTime
        self.savedTime = now
        if (text is not None) and self.enabled:
            print("{} {:.20f}".format(text, deltaTime))
        return deltaTime
    
inference_latency_util = InferenceLatencyUtil()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, 
                               kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, 
                               kernel_size = 3, stride = 1, padding = 1)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        inference_latency_util.time_flag1()
        x = self.conv1(x)
        inference_latency_util.time_flag2("Latency of conv1: ")
        x = self.relu(x)
        x = self.max_pool2d(x)
        inference_latency_util.time_flag1()
        x = self.conv2(x)
        inference_latency_util.time_flag2("Latency of conv2: ")
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = x.reshape(x.size(0),-1)
        inference_latency_util.time_flag1()
        x = self.fc1(x)
        inference_latency_util.time_flag2("Latency of fc1: ")
        x = self.relu(x)
        inference_latency_util.time_flag1()
        x = self.fc2(x)
        inference_latency_util.time_flag2("Latency of fc2: ")
        return F.log_softmax(x)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.2, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # model = Net().to(device)
    # input_data = torch.randn(1, 1, 28, 28)
    # inference_latency_util.enabled = True
    # model(input_data)
    # return

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


    #scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        #scheduler.step()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
