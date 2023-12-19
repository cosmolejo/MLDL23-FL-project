import copy
import pprint
import math

from collections import OrderedDict
import numpy as np
import torch


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients  # we do this in main, train test split
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())



    def select_clients(self):
        '''
        This method returns an array with the selected clients for the current round
        The way selection is done is by only considering the min number between
        a pre-set value for num_clients arbitraily and then chose min 
        '''
        if self.args.client_select == 0:
            num_clients = min(self.args.clients_per_round, len(self.train_clients))
            sel_clients = np.random.choice(self.train_clients, num_clients, replace=False)
        
        elif self.args.client_select == 1:  
            num_clients = min(self.args.clients_per_round, len(self.train_clients))
            """
            with 10% of clients being selected with probability 0.5 at each round
            with 30% of clients being selected with probability 0.0001 at each round
            """
            n10perc = math.ceil(len(self.train_clients)*0.1)
            list_client10 =  self.train_clients[:n10perc]
            list_client90 = self.train_clients[n10perc:]
            list_p10 = [1/n10perc] * n10perc
            list_p90 = [1/(len(self.train_clients)-n10perc)] * (len(self.train_clients)-n10perc)
            sel_clients = []
            i=0
            while i != (num_clients):
                if np.random.random() < 0.5:
                    c = np.random.choice(list_client10, p=list_p10, replace=False)
                    if c not in sel_clients:
                        sel_clients.append(c)
                        i+=1
                else:
                    c = np.random.choice(list_client90, p=list_p90, replace=False)
                    if c not in sel_clients:
                        sel_clients.append(c)
                        i+=1
            
            print(f'len clients:{len(sel_clients)}')
            print(f'selected client: {sel_clients}')
        return sel_clients

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
            this method: 1.inovkes the trainning in the client
            2. saves the output parameters for each client in a list of dicitionaries
            3. outputs this list "updates"
        """
        # suppposed to keep the weights per each client 
        # in essence this is a list of dicitionaries 
        updates = []
        for i, c in enumerate(clients):
            # our addition
            # this initialized the method "train" in client
            # which outputs model.state_dic 
            # which has as keys 'layer_weights':
            # layer_bias: 
            num_samples,client_update = c.train()
            updates.append((num_samples,copy.deepcopy(client_update))) #deep copy to not change the original dictionary of client
        return updates

    def aggregate(self, updates):
        """
        # our addition
        """
        """"
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
    
        """
        if len(updates) == 0:
            # the original model
            return self.model.state_dict()  # No updates to aggregate

        # Aggregate the model parameters using Federated Averaging
        # sets a single dic for all clients as a global dic 
        aggregated_params = OrderedDict()
        # loops throught the updates list and takes the "mean" for each key for each element

        averaged_state_dict = updates[0].copy()

        # Iterate over the layers in the state dictionaries
        for layer_name in averaged_state_dict.keys():
            # Iterate over the state dictionaries
            for state_dict in updates[1:]:
                # Add the weights of the corresponding layer
                averaged_state_dict[layer_name] += state_dict[layer_name]

            # Average the weights
            averaged_state_dict[layer_name] /= len(updates)
        """
            
        total_client_sample = 0.
        base = OrderedDict()
        for (client_samples, client_model) in updates:
            total_client_sample += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += (client_samples * value.type(torch.FloatTensor))
                else:
                    base[key] = (client_samples * value.type(torch.FloatTensor))
                

        averaged_soln = copy.deepcopy(self.model.state_dict())
        for key, value in base.items():
            if total_client_sample != 0:
                averaged_soln[key] = value.cuda() / total_client_sample
        return averaged_soln
    
    
    def update_clients_model(self,aggregated_params):
        for i, c in enumerate(self.train_clients):
            c.model.load_state_dict(aggregated_params)
        
    

    def train(self):
        '''
        This method does the "global trainning"
        It calls every method within server - basically the main of server
        Then, this method it's called in "main"
        '''

        for r in range(self.args.num_rounds):
            # our addition
            # take selected clients
            sel_clients = self.select_clients()
            if r != 0:
                self.update_clients_model(aggregated_params=aggregated_params)
            print(f"Round {r + 1}/{self.args.num_rounds}")

            # Train the model on the selected clients 
            # and ouputs "updates" the list with state_dic

            train_sel_c = self.train_round(sel_clients)

            # Aggregate the updates using FedAvg for the selected clients
            # returns 1 dicitionary with the "final" parameters of the round
            aggregated_params = self.aggregate(train_sel_c)

            # Update the global model with the aggregated parameters
            # we call the method model.load_state_dict from the "module" class

            self.model.load_state_dict(aggregated_params)
            
            # Evaluate on the train clients
            train_accuracy = self.eval_train(sel_clients,aggregated_params)
            print(f"Train Accuracy for round {r + 1} is : {train_accuracy:.4f}")

            # Test on the test clients
            test_accuracy = self.test(aggregated_params)
            print(f"Test Accuracy for round {r + 1}: {test_accuracy:.4f}")

    def eval_train(self, clients, aggregated_params):
        """
        This method handles the evaluation on the train clients
        """
        # our addition
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for client in clients:
                client.model.load_state_dict(aggregated_params)
                client_samples, client_correct = client.test(self.metrics, 'eval_train')
                total_correct += client_correct
                total_samples += client_samples
        accuracy = total_correct / total_samples
        return accuracy

    def test(self,aggregated_params):
        """
        This method handles the evaluation of the test_clients
        """
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for client in self.test_clients:  # we don't select for test we run it on all
                client.model.load_state_dict(aggregated_params)
                client_samples, client_correct = client.test(self.metrics, 'test')
                total_correct += client_correct
                total_samples += client_samples
        accuracy = total_correct / total_samples
        return accuracy
