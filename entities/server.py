import copy
from collections import OrderedDict

import numpy as np
import torch


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        for i, c in enumerate(clients):
            client_update = c.train(self.model)
            updates.append(client_update)

        return updates

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        if len(updates) == 0:
                    return self.model.state_dict()  # No updates to aggregate
        
        # Aggregate the model parameters using Federated Averaging
        aggregated_params = {}
        for param_name in updates[0].keys():
            aggregated_params[param_name] = torch.stack([update[param_name] for update in updates]).mean(dim=0)

        return aggregated_params
    
    def update_model(self, aggregated_params):
        """
        Update the global model with the aggregated parameters.
        :param aggregated_params: aggregated parameters from clients
        """
        self.model.load_state_dict(aggregated_params)

    def train(self):

        for r in range(self.args.num_rounds):
            
            print(f"Round {r+1}/{self.args.num_rounds}")
            
            # Train the model on the clients and get updates
            updates = self.train_round(self.train_clients)
            
            # Aggregate the updates using FedAvg
            aggregated_params = self.aggregate(updates)
            
            # Update the global model with the aggregated parameters
            self.update_model(aggregated_params)
            
            # Evaluate on the train clients
            train_accuracy = self.eval_train(self.train_clients)
            print(f"Train Accuracy: {train_accuracy:.4f}")

            # Test on the test clients
            test_accuracy = self.test(self.test_clients)
            print(f"Test Accuracy: {test_accuracy:.4f}")


    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for client in self.train_clients:
                client_samples, client_correct = client.evaluate(self.model)
                total_correct += client_correct
                total_samples += client_samples
        accuracy = total_correct / total_samples
        return accuracy
    


    def test(self):
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for client in self.test_clients:
                client_samples, client_correct = client.evaluate(self.model)
                total_correct += client_correct
                total_samples += client_samples
        accuracy = total_correct / total_samples
        return accuracy
        raise NotImplementedError
