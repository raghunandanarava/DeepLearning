from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode

        if mode == 'train':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ColorJitter(contrast=0.1, brightness=0.2),
                #tv.transforms.RandomHorizontalFlip(),
                #tv.transforms.RandomResizedCrop(300, scale=(0.9,1.1)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std),
            ])
        elif mode == 'val':
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                #tv.transforms.Resize(256),
                #tv.transforms.CenterCrop(384),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = imread(self.data['filename'][index])
        image = gray2rgb(image)



        label = np.array((self.data['crack'][index], self.data['inactive'][index]), dtype='float32')

        #do the transform
        if self._transform:
            image = self._transform(image)

        # if self.mode == 'train':
        #     label[0] = torch.cat([(label[0] >= 0.5) * 1.0, (label[0] < 0.5) * 1.0])
        #     label[1] = torch.cat([(label[1] >= 0.5) * 1.0, (label[1] < 0.5) * 1.0])

        return image, label
