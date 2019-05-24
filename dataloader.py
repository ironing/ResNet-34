import os
import numpy as np
import torch.utils.data as data
from PIL import Image

def make_dataset(data_dir):
    lblarr_train = []
    # lblarr_test = []
    train_data = []
    # test_data = []

    train_csv = 'ISIC_2019_train.csv'
    fn = os.path(data_dir, train_csv)
    print('Loading train from {} '.format(fn))
    file = open(fn, 'r')
    for line in file:
        line = line.split(',')
        item = (line[0], int(line.index('1')))
        lblarr_train.append(line.index('1'))
        train_data.append(item)
    file.close()
    unique_labels = np.unique(lblarr_train).tolist()

    return train_data, unique_labels

class dataloader(data.Dataset):
    def __init__(self, train=True, transform=None, data_dir='../data/ISIC2019/'):
        self.train_data, self.classes = make_dataset(data_dir=data_dir)
        self.transform = transform
        self.train = train # training set or test set

        train_data = 'ISIC_2019_Training_Input'
        self.train_data_dir = os.path.join(data_dir, train_data)

    def __getitem__(self, index):
        """
               Args:
                   index (int): Index
               Returns:
                   tuple: (sample, target) where target is class_index of the target class.
               """
        #if self.train:
        path, target = self.train_data[index]
        #else:
            #path, target = self.test_data[index]

        path = os.path.join(self.train_data_dir, path)
        path += '.jpg'
        imagedata = default_loader(path)
        if self.transform is not None:
            imagedata = self.transform(imagedata)
        return imagedata, target

    def __len__(self):
        #if self.train:
        return len(self.train_data)
        #else:
            #return len(self.test_data)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
