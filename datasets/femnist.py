import numpy as np
import os
import sys
path = os.getcwd()
if 'kaggle' not in path:
    import datasets.ss_transforms as tr
else:
    sys.path.append('datasets')
    import ss_transforms as tr



#from torchvision import transforms

#from torch import from_numpy
from PIL import Image
IMAGE_SIZE = 28

from torch.utils.data import Dataset

IMAGE_SIZE = 28

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

#convert_tensor = transforms.ToTensor()


class Femnist(Dataset):

    def __init__(self,
                 data: dict,
                 transform: tr.Compose,
                 client_name: str):
        super().__init__()
        self.samples = [(image, label) for image, label in zip(data['x'], data['y'])]
        self.transform = transform
        self.client_name = client_name

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.fromarray(np.uint8(np.array(sample[0]).reshape(28, 28) * 255))
        label = sample[1]

        if self.transform is not None:
            image = self.transform(np.array(image))

        return image, label

    def __len__(self) -> int:
        return len(self.samples)
