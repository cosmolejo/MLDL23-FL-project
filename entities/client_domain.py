import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
import threading
import torch.nn.functional as F
import torch.distributions as distributions
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
        self.z_dim = int(3136)
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs,
                                       shuffle=True) if not test_client else None  # ,drop_last=True
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.len_dataset = len(self.dataset)
        self.pk = None
        self.r_mu = nn.Parameter(torch.zeros(args.num_classes, int(self.z_dim / 2))).cuda()
        self.r_sigma = nn.Parameter(torch.ones(args.num_classes, int(self.z_dim / 2))).cuda()
        self.C = nn.Parameter(torch.ones([]))

    def __str__(self):
        return self.idx


    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError

    def featurize(self, x, num_samples=1, return_dist=False):
        # print('Im in featurize 1')
        # print(x.shape)
        self.model = self.model.cuda()
        features = self.model(x)
        z_mu = features[:, :int(self.z_dim / 2)]
        z_sigma = F.softplus(features[:, int(self.z_dim / 2):])
        z_mu = z_mu.to(x.device)
        # print('printing z_mu and z_sigma')
        # print(z_mu.shape , z_sigma.shape)
        z_sigma = z_sigma.to(x.device)
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z = z_dist.rsample([num_samples])
        # print("z size before view:", z.size())
        z = z.view([-1, int(self.z_dim / 2)])
        # print("z size after view:", z.size())
        # print('Im in featurize 2')
        # print(z.shape)
        if return_dist:
            return z, (z_mu, z_sigma)
        else:
            return z

    def classify(self, z):
        # print('im in classify')
        # print(z.shape)
        fc1 = nn.Linear(7 * 7 * 32, 2048).to(z.device)
        fc2 = nn.Linear(2048, self.args.num_classes).to(z.device)
        x = F.relu(fc1(z))
        x = fc2(x)
        # print('im in classify')
        # print(x.shape)
        return x

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

            # outputs = self.model(images)
            z, (z_mu, z_sigma) = self.featurize(images, return_dist=True)
            logits = self.classify(z)
            # print('im after logits')
            # print(logits.shape , labels.shape)
            loss = self.criterion(logits, labels)

            obj = loss
            regL2R = torch.zeros_like(obj)
            regCMI = torch.zeros_like(obj)
            # if self.args.L2R_coeff != 0.0:
            regL2R = z.norm(dim=1).mean()
            obj = obj + 0.001 * regL2R  # remember to put L2R coefficient as argument
            # if self.args.CMI_coeff != 0.0:
            self.r_sigma = self.r_sigma.cuda()
            self.r_mu = self.r_mu.cuda()
            r_sigma_softplus = F.softplus(self.r_sigma)
            r_mu = self.r_mu[labels]
            r_sigma = r_sigma_softplus[labels]
            z_mu_scaled = z_mu * self.C
            z_sigma_scaled = z_sigma * self.C
            regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + (
                        z_sigma_scaled ** 2 + (z_mu_scaled - r_mu) ** 2) / (2 * r_sigma ** 2) - 0.5
            regCMI = regCMI.sum(1).mean()
            obj = obj + 0.0001 * regCMI  # remember to put CMI coefficient as argument

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            i += 1
            running_loss += obj.item()
            # print(labels.shape)
            # accuracy = (logits.argmax(1)==labels).float().mean()

            correct_predictions = torch.sum(torch.eq(logits.argmax(1), labels)).item()
            tot_correct_predictions += correct_predictions

        loss_for_this_epoch = running_loss / i
        accuracy = tot_correct_predictions / self.len_dataset * 100
        return loss_for_this_epoch, accuracy

    def train(self, r):
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
            if r > self.args.num_rounds * 0.7:
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

    def test(self):
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

                features = self.model(images)
                z = self.featurize(features)
                logits = self.classify(z)
                # from the logits we get the actual class probabilities

                _, predicted = torch.max(logits.data, 1)

                total += labels.size(0)
                correct += torch.eq(predicted, labels).sum().item()

        return total, correct

    def get_pk(self):
        return self.pk

    def set_pk(self, total_train_data):
        self.pk = len(self.train_loader) / total_train_data

    def get_total_train(self):
        return len(self.train_loader)
