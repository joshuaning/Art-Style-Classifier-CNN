import os
from os import walk
import copy
import pickle

import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch.optim as optim

if __name__ == '__main__':
    # check if files can be opened
    print(os.listdir("./"))

    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only. If you want to enable GPU, please to go Edit > Notebook Settings > Hardware Accelerator and select GPU.")

    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize([0.5162, 0.4644, 0.3975], 
                                                        [0.2728, 0.2641, 0.2575])])
    dataset = datasets.ImageFolder("/home/joshning/data_256/art_pictures/train",
                                transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, 
                                            shuffle=True, num_workers=4)


    # implementation of VGG-BN for 3*256*256 inputs
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=(1,1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True))
            
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=(1,1)),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True)
            )
            
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1, ceil_mode=False)
            
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=(1,1)),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True))
            
            self.conv4 = nn.Sequential(
                nn.Conv2d(128, 128, 3, padding=(1,1)),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace = True))
            
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1, ceil_mode=False)

            self.conv5 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=(1,1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True))
            
            self.conv6 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=(1,1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True))
            
            self.conv7 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=(1,1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True))
            
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1, ceil_mode=False)
            
            self.conv8 = nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=(1,1)),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True))
            
            self.conv9 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=(1,1)),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True))
            
            self.conv10 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=(1,1)),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True))
            
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1, ceil_mode=False)

            self.conv11 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=(1,1)),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True))
            
            self.conv12 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=(1,1)),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True))
            
            self.conv13 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=(1,1)),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace = True))
            
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1, ceil_mode=False)

            
            self.adaptive = nn.AdaptiveAvgPool2d(output_size=(7,7))
            
            self.fc1 = nn.Sequential(
                nn.Linear(in_features=25088, out_features = 4096, bias = True),
                nn.ReLU(inplace = True),
                nn.Dropout(p=0.3, inplace = False))
            
            self.fc2 = nn.Sequential(
                nn.Linear(in_features=4096, out_features = 4096, bias = True),
                nn.ReLU(inplace = True),
                nn.Dropout(p=0.3, inplace = False))
            
            self.fc3 = nn.Linear(in_features=4096, out_features = 10, bias = True)
            
        def forward(self, x):
            x = self.conv1(x) 
            x = self.conv2(x) 
            x = self.pool1(x) 
            x = self.conv3(x) 
            x = self.conv4(x) 
            x = self.pool2(x) 
            x = self.conv5(x) 
            x = self.conv6(x)   
            x = self.conv7(x)  
            x = self.pool3(x) 
            x = self.conv8(x) 
            x = self.conv9(x) 
            x = self.conv10(x)
            x = self.pool4(x)
            x = self.conv11(x)
            x = self.conv12(x)
            x = self.conv13(x)
            x = self.pool5(x)
            x = self.adaptive(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x

    net = Net()
    net.to(device)

    summary(net, input_size = (32, 3, 256, 256))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    num_epoch = 50
    total_step = len(dataloader)
    stats = np.empty((0,4), float)


    for epoch in tqdm(range(num_epoch)):  # loop over the dataset multiple times
        running_loss = 0.0
        running_correct = 0
        print('------ Starting Epoch [{}/{}] ------'.format(epoch+1, num_epoch))
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            
            # back prop
            loss.backward()
            optimizer.step()

            # stats
            running_loss += loss.item()*inputs.size(0)
            curr_correct = torch.sum(preds == labels)
            running_correct += curr_correct
            stats = np.vstack((stats, np.array([curr_correct, running_loss, running_correct, (i+1)*32])))
            print('Epoch [{}/{}], Step [{}/{}], Running Loss: {:.4f}, running Correct: [{}/{}]'
                .format(epoch+1, num_epoch, i+1, total_step, running_loss, running_correct, (i+1)*32))
        
        #save trained model every epoch
        PATH = '/scratch/eecs351w23_class_root/eecs351w23_class/joshning/epoch_{}.pth'.format(epoch+1)
        torch.save(net.state_dict(), PATH)
        
    print('Finished Training')

    import pandas as pd
    DF = pd.DataFrame(stats)
    DF.to_csv("./stats/vgg_16_stats.csv")