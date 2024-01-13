import sys
import os
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import ConcatDataset
import json
from tqdm import tqdm

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


def read_femnist_dir(data_dir):
    data = defaultdict(lambda: {})
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in tqdm(files):
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        data.update(cdata['user_data'])
    return data


def read_femnist_data(train_data_dir, test_data_dir=None, args=None):
    """
    If only one directory was given, the function returns the
    all_data folder content
    """
    if test_data_dir:
        if args.rotation:
            data = read_femnist_dir(train_data_dir)
            data_test = read_femnist_dir(test_data_dir)
            for key, val in data_test.items():
                data[key] = val
            return data
        return read_femnist_dir(train_data_dir), read_femnist_dir(test_data_dir)
    else:
        return read_femnist_dir(train_data_dir)


def get_transforms(args):
    # TODO: test your data augmentation by changing the transforms here!
    if not args.rotation:

        train_transforms = sstr.Compose([transforms.ToTensor(), nptr.Normalize((0.5,), (0.5,)), ])
        test_transforms = sstr.Compose([transforms.ToTensor(), nptr.Normalize((0.5,), (0.5,)), ])

        return train_transforms, test_transforms
    else:
        rotations = [0, 15, 30, 45, 60, 75]
        rot_transforms = {}
        for rot in rotations:
            rot_transforms[f'{rot}'] = sstr.Compose([
                transforms.ToTensor(),
                nptr.Normalize((0.5,), (0.5,)),
                sstr.Rotation(rot)
            ])
        return rot_transforms


def get_datasets(args):

    if not args.rotation:
        train_transforms, test_transforms = get_transforms(args)
    else:
        rot_transforms = get_transforms(args)

    train_datasets, test_datasets = [], []

    if args.federated:
        # elif args.dataset == 'femnist':
        niid = args.niid
        train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'train')
        test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if niid else 'iid', 'test')

        if args.rotation:
            train_data = read_femnist_data(train_data_dir, test_data_dir, args)
        else:
            train_data, test_data = read_femnist_data(train_data_dir, test_data_dir)

        if args.rotation:
            train_rotations = {
                '0':  [],
                '15': [],
                '30': [],
                '45': [],
                '60': [],
                '75': []
            }
            count = 0
            for user, data in train_data.items():
                if count < 1000:

                    if count <= 170:
                        train_rotations['0'].append(Femnist(data, rot_transforms['0'], user))
                    elif 170 < count <= 336:
                        train_rotations['15'].append(Femnist(data, rot_transforms['15'], user))
                    elif 336 < count <= 502:
                        train_rotations['30'].append(Femnist(data, rot_transforms['30'], user))
                    elif 502 < count <= 668:
                        train_rotations['45'].append(Femnist(data, rot_transforms['45'], user))
                    elif 668 < count <= 834:
                        train_rotations['60'].append(Femnist(data, rot_transforms['60'], user))
                    elif 834 < count:
                        train_rotations['75'].append(Femnist(data, rot_transforms['75'], user))
                    count += 1
                else:
                    break
            return train_rotations
        else:
            for user, data in train_data.items():
                train_datasets.append(Femnist(data, train_transforms, user))
            for user, data in test_data.items():
                test_datasets.append(Femnist(data, test_transforms, user))

        return train_datasets, test_datasets
    else:
        all_data_dir = os.path.join('data', 'femnist', 'data', 'all_data')
        all_data = read_femnist_data(all_data_dir)
        centralized_datasets = []
        if args.rotation:
            train_rotations = {
                '0': [],
                '15': [],
                '30': [],
                '45': [],
                '60': [],
                '75': []
            }
            count = 0
            for user, data in all_data.items():
                if count < 1000:

                    if count <= 170:
                        train_rotations['0'].append(Femnist(data, rot_transforms['0'], ''))
                    elif 170 < count <= 336:
                        train_rotations['15'].append(Femnist(data, rot_transforms['15'], ''))
                    elif 336 < count <= 502:
                        train_rotations['30'].append(Femnist(data, rot_transforms['30'], ''))
                    elif 502 < count <= 668:
                        train_rotations['45'].append(Femnist(data, rot_transforms['45'], ''))
                    elif 668 < count <= 834:
                        train_rotations['60'].append(Femnist(data, rot_transforms['60'], ''))
                    elif 834 < count:
                        train_rotations['75'].append(Femnist(data, rot_transforms['75'], ''))
                    count += 1
                else:
                    break
            return train_rotations
        else:
            for user, data in all_data.items():
                centralized_datasets.append(Femnist(data, train_transforms, ''))
            return ConcatDataset(centralized_datasets)

