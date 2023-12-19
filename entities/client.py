import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
import threading

from utils.utils import HardNegativeMining, MeanReduction


class Client:

    def __init__(self, args, dataset, model, optimizer, idx, test_client=False):
        """
        putting the optimizer as an input parameter
        """
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.idx = idx
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True,
                                       drop_last=True) if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.len_dataset = len(self.dataset)

    def __str__(self):
        return self.idx

    @staticmethod
    def update_metric(metric, outputs, labels, key):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric[key].update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError

    def run_epoch(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        # There is also scheduler for the learning rate that we will put later.
        # self.optim_scheduler.step()
        tot_correct_predictions = 0
        running_loss = 0.0
        i = 0
        for cur_step, (images, labels) in enumerate(self.train_loader):
            images = images.cuda()
            labels = labels.cuda()

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)
            

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()
            i +=1
            
            predictions = torch.argmax(outputs, dim=1)

            correct_predictions = torch.sum(torch.eq(predictions, labels)).item()
            tot_correct_predictions += correct_predictions

        loss_for_this_epoch = running_loss / i
        accuracy = tot_correct_predictions / self.len_dataset * 100
        return loss_for_this_epoch, accuracy
       
        """
        criterion = nn.CrossEntropyLoss().to(self.device)
        if self.optim == 'adam':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        losses = np.empty(num_epochs)

        for epoch in range(num_epochs):
            self.model.train()
            if self.mixup:
                losses[epoch] = self.run_epoch_with_mixup(optimizer, criterion)
            else:
                losses[epoch] = self.run_epoch(optimizer, criterion)

        self.losses = losses
        update = self.model.state_dict()
        return self.num_train_samples, update
        

        """
        """
        def run_epoch(self, optimizer, criterion):
            running_loss = 0.0
            i = 0
            for j, data in enumerate(self.trainloader):
                input_data_tensor, target_data_tensor = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(input_data_tensor)
                loss = criterion(outputs, target_data_tensor)
                loss.backward()  # gradient inside the optimizer (memory usage increases here)
                running_loss += loss.item()
                optimizer.step()  # update of weights
                i += 1
            if i == 0:
                print("Not running epoch", self.id)
                return 0
            return running_loss / i
        """

       

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        # initial_model_params = copy.deepcopy(self.model.state_dict())
        # maybe it is needed
        

        for epoch in range(self.args.num_epochs):
            print(f"tid={str(threading.get_ident())[-7:]} - k_id={self.idx}: START EPOCH={epoch + 1}/{self.args.num_epochs}")
            
            loss_each_epoch, train_accuracy = self.run_epoch()
            
            if epoch != self.args.num_epochs:
                print(f"tid={str(threading.get_ident())[-7:]} - k_id={self.idx}: END   EPOCH={epoch + 1}/{self.args.num_epochs} - ",end="")
                print(f"Loss={round(loss_each_epoch, 3)}, Accuracy={round(train_accuracy, 2)}%")
            else:
                last_epoch_loss = loss_each_epoch
                print(f"tid={str(threading.get_ident())[-7:]} - k_id={self.idx}: END   EPOCH={epoch + 1}/{self.args.num_epochs} - ",end="")
                print(f"Loss last epochs={round(last_epoch_loss, 3)}, Accuracy={round(train_accuracy, 2)}%")
            

        return (len(self.train_loader),self.model.state_dict()) 

    def test(self, metric, key):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.cuda()
                labels = labels.cuda()

                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += torch.eq(predicted, labels).sum().item()

                self.update_metric(metric, outputs, labels, key)
        return total, correct
