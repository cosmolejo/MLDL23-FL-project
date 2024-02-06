import json
import os
import sys

import numpy as np
import pandas as pd
import torch
torch.set_warn_always(False)
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms

path = os.getcwd()
if 'kaggle' not in path:
    from datasets.femnist import Femnist
else:
    sys.path.append('datasets')
    from femnist import Femnist

IMAGE_SIZE = 28


class Centralized:

    def __init__(self, data, model, args, angle=None, data_test_loo=None):
        self.data = data
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.SGD(model.parameters(),
                                         lr=args.lr, momentum=args.m,
                                         weight_decay=args.wd)  # define loss function criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        if angle:
            self.angle = angle
            self.data_test = data_test_loo

    def n_classes(self, batch):
        return batch['class'].unique().shape[0]

    def get_data(self):
        df = pd.DataFrame()
        print('loading files.....')
        for dirname, _, filenames in os.walk(self.path):
            for filename in filenames:
                # print(filename)
                data = json.load(open(os.path.join(dirname, filename)))

                temp_df = pd.DataFrame(data['user_data'])
                temp_df = temp_df.reset_index(drop=True)
                df = pd.concat([df, temp_df], axis=1)  # ignore_index=True
        df = df.rename(index={0: "x", 1: "y"})
        return df

    def train_test_tensors_rot_ng(self, datasets):
        if self.args.loo:

            train_subset = self.data #for dataset_list in self.data.values() for dataset in dataset_list
            test_subset = self.data_test#values() for dataset in dataset_list
        else:
            if self.args.rotation:
                datasets = ConcatDataset([dataset for dataset_list in datasets.values() for dataset in dataset_list])
            # receive a tuple of objects and split in train and test
            train_size = int(0.8 * len(datasets))
            test_size = len(datasets) - train_size
            # Create random train/test splits
            train_subset, test_subset = random_split(datasets, [train_size, test_size])
        return train_subset, test_subset

    def training(self, torch_train):

        train_loader = DataLoader(torch_train, batch_size=self.args.bs, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loss_avg = []
        for epoch in range(self.args.num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            train_loss_avg.append(running_loss)
        print('Finished Training')
        return train_loss_avg

    def accuracy_of_model(self, val_loader):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct // total
        print(f'Accuracy of the network on the 10000 test images: {accuracy} %')
        return accuracy

    def pipeline(self):

        torch_train, torch_test = self.train_test_tensors_rot_ng(self.data)
        print('Training')
        train_loss_avg = self.training(torch_train)
        print('Done.')
        # printing accuracy
        val_loader = DataLoader(torch_test, batch_size=self.args.bs, shuffle=False)
        print('Validating')
        accuracy = self.accuracy_of_model(val_loader)

        train_dict = {'Epochs': np.array(range(self.args.num_epochs)),
                      'Train Loss Average': np.array(train_loss_avg),
                      'Test accuracy': np.array(accuracy)}
        train_csv = pd.DataFrame(train_dict)
        if self.args.loo:
            train_csv.to_csv(
                f'FedAVG_lr:{self.args.lr}_mom:{self.args.m}_epochs:{self.args.num_epochs}_bs:{self.args.bs}_seed:{self.args.seed}_angle:{self.angle}.csv',
                index=False)
        else:
            train_csv.to_csv(
                f'FedAVG_lr:{self.args.lr}_mom:{self.args.m}_epochs:{self.args.num_epochs}_bs:{self.args.bs}_seed:{self.args.seed}.csv',
                index=False)

        # print('Summary')
        # print(summary(self.model))
