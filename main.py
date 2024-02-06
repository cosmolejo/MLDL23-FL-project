import random
import sys

import numpy as np
from torch import nn
from torchvision.models import resnet18

sys.path.append('datasets')

from entities.client import Client
from entities.client_domain import Client as Client_domain

from entities.centralized import Centralized

from entities.server import Server

from utils.args import get_parser

from models.cnn import CNN
from models.cnn_domain import CNN as CNN_domain
from utils.data_generation import *


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_num_classes(dataset):
    if dataset == 'femnist':
        return 62
    else:
        raise NotImplementedError


def model_init(args):
    """
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    """
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model

    if args.model == 'cnn':
        if args.fedSR:
            model = CNN_domain(get_dataset_num_classes('femnist'))
        else:
            model = CNN(get_dataset_num_classes('femnist'))
        return model
    else:
        raise NotImplementedError


###########################################
## Moved read_femnist_dir, read_femnist_data,
## get_transforms and get_datasets into utils.utils
###########################################


def gen_clients(args, train_datasets, test_datasets, model):
    clients = [[], []]
    # define loss function criterion = nn.CrossEntropyLoss()
    idx = 0
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model,
                                     optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                     idx=idx, test_client=i == 1)
                              )
            idx += 1

    total_train_data = 0
    for c in clients[0]:
        total_train_data += c.get_total_train()
    for c in clients[0]:
        c.set_pk(total_train_data)

    print(f'Clients: Train {len(clients[0])}, Test {len(clients[1])}')
    return clients[0], clients[1]


def gen_rot_clients(args, datasets, model, angle=None):
    print('Splitting Data')
    idx = 0
    clients = [[], []]
    if args.loo:
        if not args.federated:
            train = []
            test = []
            for key in datasets.keys():
                if key == angle:
                    for ds in datasets[key]:
                        test.append(ds)
                else:
                    for ds in datasets[key]:
                        train.append(ds)

            print('Done')
            train = ConcatDataset([dataset for dataset in train])
            test = ConcatDataset([dataset for dataset in test])
            return train, test

        else:
            print('datasets: ', len(datasets.values()), 'keys:', datasets.keys())
            for key in datasets.keys():
                if key == angle:
                    for ds in datasets[key]:
                        clients[1].append(Client(args, ds, model,
                                                 optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                                 idx=idx, test_client=True)
                                          )
                        idx += 1
                else:
                    for ds in datasets[key]:
                        clients[0].append(Client(args, ds, model,
                                                 optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                                 idx=idx, test_client=False)
                                          )
                        idx += 1
    else:
        indices = list(range(1000))
        sample = random.sample(indices, 700)
        split = [False if i not in sample else True for i in indices]

        for key in datasets.keys():
            for ds in datasets[key]:
                if split[idx]:
                    clients[0].append(Client(args, ds, model,
                                             optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                             idx=idx, test_client=False)
                                      )
                else:
                    clients[1].append(Client(args, ds, model,
                                             optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                             idx=idx, test_client=True)
                                      )
                idx += 1
    print('Done')
    return clients[0], clients[1]


def gen_rot_clients_fedsr(args, datasets, model, angle=None):
    idx = 0
    clients = [[], []]
    if args.loo:

        for key in datasets.keys():
            if key == angle:
                for ds in datasets[key]:
                    clients[1].append(Client_domain(args, ds, model,
                                                    optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                                    idx=idx, test_client=True)
                                      )
                    idx += 1
            else:
                for ds in datasets[key]:
                    clients[0].append(Client_domain(args, ds, model,
                                                    optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                                    idx=idx, test_client=False)
                                      )
                    idx += 1
    else:
        indices = list(range(1000))
        sample = random.sample(indices, 700)
        split = [False if i not in sample else True for i in indices]

        for key in datasets.keys():
            for ds in datasets[key]:
                if split[idx]:
                    clients[0].append(Client_domain(args, ds, model,
                                                    optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                                    idx=idx, test_client=False)
                                      )
                else:
                    clients[1].append(Client_domain(args, ds, model,
                                                    optimizer=torch.optim.SGD(model.parameters(), lr=args.lr),
                                                    idx=idx, test_client=True)
                                      )
                idx += 1
    print(f'Clients len {len(clients)}, train {len(clients[0])}, test {len(clients[1])}')
    return clients[0], clients[1]


def fed_exec(args, model, rot_dataset=None, angle=None, train_datasets=None, test_datasets=None):
    print('Gererating clients...')
    if args.rotation:
        if args.loo:
            if args.fedSR:
                train_clients, test_clients = gen_rot_clients_fedsr(args, rot_dataset, model, angle)
            else:
                train_clients, test_clients = gen_rot_clients(args, rot_dataset, model, angle)

        else:
            if args.fedSR:
                train_clients, test_clients = gen_rot_clients_fedsr(args, rot_dataset, model)
            else:
                train_clients, test_clients = gen_rot_clients(args, rot_dataset, model)

    else:
        train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
    print('Done.')
    print('Creating server')
    server = Server(args, train_clients, test_clients, model, angle)
    print('Training start.....')
    server.train()


def centralized_exec(args, model, angle=None, rot_dataset=None):
    if args.loo:
        train_datasets, test_datasets = gen_rot_clients(args, rot_dataset, model, angle)
    else:
        print('Generate datasets...')
        centralized_dataset = get_datasets(args)
        print('Done.')
    print('Creating centralized session')
    if args.loo:
        centralized = Centralized(data=train_datasets, model=model, args=args,
                                  angle=angle, data_test_loo=test_datasets)
    else:
        centralized = Centralized(data=centralized_dataset, model=model, args=args)
    print('Training start.....')
    centralized.pipeline()


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    print('Initializing model...')
    model = model_init(args)
    model.cuda()
    print('Done.')

    if args.federated:
        print('Generate datasets...')
        if args.rotation:
            rot_dataset = get_datasets(args)
        else:
            train_datasets, test_datasets = get_datasets(args)
        print('Done.')

        if args.loo:
            angles = ['0', '15', '30', '45', '60', '75']
            for a in angles:
                print('Training Domain for angle:', a)
                fed_exec(args, model, rot_dataset=rot_dataset, angle=a)
        else:
            if args.rotation:
                fed_exec(args, model, rot_dataset=rot_dataset)
            else:
                fed_exec(args, model, train_datasets=train_datasets, test_datasets=test_datasets)
    else:
        if args.loo:
            print('Generate datasets...')
            rot_dataset = get_datasets(args)
            print('Done.')

            angles = ['0', '15', '30', '45', '60', '75']
            for a in angles:
                print('Training Domain for angle:', a)
                centralized_exec(args, model, rot_dataset=rot_dataset, angle=a)
        else:
            centralized_exec(args, model)


if __name__ == '__main__':
    path = os.getcwd()
    if 'kaggle' not in path:
        import datasets.ss_transforms as sstr
        import datasets.np_transforms as nptr
        from datasets.femnist import Femnist
    else:
        sys.path.append('datasets')
        import ss_transforms as sstr
        import np_transforms as nptr
        from femnist import Femnist

    main()
