import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
import threading

from utils.utils import HardNegativeMining, MeanReduction
import torch.nn.utils.prune as prune


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
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs,
                                       shuffle=True) if not test_client else None  # ,drop_last=True
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.len_dataset = len(self.dataset)
        self.pk = None

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
            i += 1

            predictions = torch.argmax(outputs, dim=1)

            correct_predictions = torch.sum(torch.eq(predictions, labels)).item()
            tot_correct_predictions += correct_predictions

        loss_for_this_epoch = running_loss / i
        accuracy = tot_correct_predictions / self.len_dataset * 100
        return loss_for_this_epoch, accuracy

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        # initial_model_params = copy.deepcopy(self.model.state_dict())
        # maybe it is needed
        sparsity = 0.0

        for epoch in range(self.args.num_epochs):
            print(
                f"tid={str(threading.get_ident())[-7:]} - k_id={self.idx}: START EPOCH={epoch + 1}/{self.args.num_epochs}")

            loss_each_epoch, train_accuracy = self.run_epoch()

            if epoch != self.args.num_epochs - 1:  # All epoch
                print(
                    f"tid={str(threading.get_ident())[-7:]} - k_id={self.idx}: END   EPOCH={epoch + 1}/{self.args.num_epochs} - ",
                    end="")
                print(f"Loss={round(loss_each_epoch, 3)}, Accuracy={round(train_accuracy, 2)}%")

            elif epoch == self.args.num_epochs - 1:  # Last epoch
                last_epoch_loss = loss_each_epoch
                print(
                    f"tid={str(threading.get_ident())[-7:]} - k_id={self.idx}: END   EPOCH={epoch + 1}/{self.args.num_epochs} - ",
                    end="")
                print(f"Loss last epochs:{round(last_epoch_loss, 3)}, Accuracy={round(train_accuracy, 2)}%")

        if self.args.prune == True:
            if self.args.conv == False and self.args.linear == False:
                raise Exception("Choose a layer to prune")

            if self.args.structured == True:
                print(f'You are using structured pruning')
                # Specify the pruning method (e.g., L1 unstructured pruning)
                if self.args.conv == True:
                    parameters_to_prune = [module for module in
                                           filter(lambda m: type(m) == torch.nn.Conv2d, self.model.modules())]
                if self.args.linear == True:
                    parameters_to_prune = [module for module in
                                           filter(lambda m: type(m) == torch.nn.Linear, self.model.modules())]
                # Apply pruning to the entire model
                for m in parameters_to_prune:
                    prune.ln_structured(m, name='weight', amount=self.args.amount_prune, n=1, dim=0)

            else:
                print(f'You are using unstructured pruning')
                # Specify the pruning method (e.g., L1 unstructured pruning)
                if self.args.conv == True:
                    parameters_to_prune = [(module, "weight") for module in
                                           filter(lambda m: type(m) == torch.nn.Conv2d, self.model.modules())]
                if self.args.linear == True:
                    parameters_to_prune = [(module, "weight") for module in
                                           filter(lambda m: type(m) == torch.nn.Linear, self.model.modules())]
                # Apply pruning to the entire model
                prune.global_unstructured(
                    parameters=parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.args.amount_prune,
                )

            sparsity = 100. * float(
                torch.sum(self.model.conv1.weight == 0) + torch.sum(self.model.conv2.weight == 0) + torch.sum(
                    self.model.fc1.weight == 0) + torch.sum(self.model.fc2.weight == 0)) / float(
                self.model.conv1.weight.nelement() + self.model.conv2.weight.nelement() + self.model.fc1.weight.nelement() + self.model.fc2.weight.nelement())

        return (len(self.train_loader), self.model.state_dict(), last_epoch_loss, sparsity)

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

    def get_pk(self):
        return self.pk

    def set_pk(self, total_train_data):
        self.pk = len(self.train_loader) / total_train_data

    def get_total_train(self):
        return len(self.train_loader)