import copy

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
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        sel_clients = np.random.choice(self.train_clients, num_clients, replace=False)
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
            #our addition
            # this initialized the method "train" in client
            # which outputs model.state_dic 
            #which has as keys 'layer_weights':
            # layer_bias: 
            client_update = c.train()
            updates.append(client_update)
        return updates

    def aggregate(self, updates):

        #our addition
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        if len(updates) == 0:
                    # the original model
                    return self.model.state_dict()  # No updates to aggregate
        
        # Aggregate the model parameters using Federated Averaging
        # sets a single dic for all clients as a global dic 
        aggregated_params = OrderedDict()
        # loops throught the updates list and takes the "mean" for each key for each element

        aggregated_params = copy.deepcopy(updates[0])
        #for key in aggregated_params:
        #    aggregated_params[key] = torch.stack([update[key].float() for update in updates]).mean(dim=0)
            # np.sum([update[key].cpu().data.numpy() for update in updates]) / len(updates)
        for key in aggregated_params:
            dtype = updates[0][key].dtype  # Get the dtype of the first update
            if dtype == torch.float32 or dtype == torch.float64 or dtype == torch.long:
                # If the dtype is float, compute the mean directly
                aggregated_params[key] = torch.stack([update[key].float() for update in updates]).mean(dim=0)
            else:
                # If the dtype is not float, convert to float first and then compute the mean
                aggregated_params[key] = torch.stack([update[key] for update in updates]).mean(dim=0)

        return aggregated_params
    


    def train(self):
        '''
        This method does the "global trainning"
        It calls every method within server - basically the main of server
        Then, this method it's called in "main"
        '''

        for r in range(self.args.num_rounds):
            #our addition
            # take selected clients
            sel_clients = self.select_clients()
            print(f"Round {r+1}/{self.args.num_rounds}")
            
            # Train the model on the selected clients 
            # and ouputs "updates" the list with state_dic
        
            train_sel_c = self.train_round(sel_clients)
            
            # Aggregate the updates using FedAvg for the selected clients
            # returns 1 dicitionary with the "final" parameters of the round
            aggregated_params = self.aggregate(train_sel_c)
            #print(f'For the round: {r+1} the aggregated parameters are: {aggregated_params}.')
            
            # Update the global model with the aggregated parameters
            # we call the method model.load_state_dict from the "module" class

            self.model.load_state_dict(aggregated_params)

            
            # Evaluate on the train clients
            train_accuracy = self.eval_train(sel_clients)
            print(f"Train Accuracy for round {r+1} is : {train_accuracy:.4f}")

            # Test on the test clients
            test_accuracy = self.test(self.clients)
            print(f"Test Accuracy for for round {r+1}: {test_accuracy:.4f}")


    def eval_train(self, clients):
        """
        This method handles the evaluation on the train clients
        """
        #our addition
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for client in clients:
                client_samples, client_correct = client.test(self.metrics)
                total_correct += client_correct
                total_samples += client_samples
        accuracy = total_correct / total_samples
        return accuracy
    


    def test(self):
        """
        This method handles the evaluation of the test_clients
        """
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for client in self.test_clients: # we don't select for test we run it on all
                client_samples, client_correct = client.test(self.metrics)
                total_correct += client_correct
                total_samples += client_samples
        accuracy = total_correct / total_samples
        return accuracy
        raise NotImplementedError
