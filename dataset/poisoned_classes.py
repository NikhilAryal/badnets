import random
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
import os 

class Trigger(object):
    def __init__(self, height, width, path, size, label):
        # self.trigger_img = trigger_img
        self.trigger_size = size 
        self.width = width
        self.height = height
        self.trigger_label = label
        
        if path is not None:
            self.trigger_img = Image.open(path).convert('RGB')
    
    def insert_trigger(self, img):
        img.paste(self.trigger_img, (self.width - self.trigger_size, self.height - self.trigger_size))
        return img
        
        
class BackDooredCIFAR10(CIFAR10):
    def __init__(self, args, root, train = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download = False):
        super().__init__(root, train, transform, target_transform, download)
        
        self.height, self.width, self.channels = self.__shape__()
        self.trigger_path = args.trigger_path
        self.poisoning_rate = args.rate if train else 1.0
        indices = range(len(self.targets))
        self.trigger_handler = Trigger(self.height, self.width, self.trigger_path, args.trigger_size, args.trigger_label)
        
        # get 1 poisoned label for 10 inputs (with 0.1)
        self.poisoned_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Dataset Poisoning: {len(self.poisoned_indices)} index poisoned over {len(indices)} indices with rate {self.poisoning_rate})")

        
    def __shape__(self):
        return self.data.shape[1:]
            
    def __getitem__(self, index):
        # return super().__getitem__(index)
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if index in self.poisoned_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.insert_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
        

class BackDooredMNIST(MNIST):
    def __init__(self, args, root, train = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download = False):
        super().__init__(root, train, transform, target_transform, download)
        
        self.height, self.width = self.__shape__()
        self.channels = 1
        self.trigger_path = args.trigger_path
        self.poisoning_rate = args.rate if train else 1.0
        indices = range(len(self.targets))
        self.trigger_handler = Trigger(self.height, self.width, self.trigger_path, args.trigger_size, args.trigger_label)
        
        # get 1 poisoned label for 10 inputs (with 0.1)
        self.poisoned_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Dataset Poisoning: {len(self.poisoned_indices)} index poisoned over {len(indices)} indices with rate {self.poisoning_rate})")

        
    def __shape__(self):
        return self.data.shape[1:]
            
    def __getitem__(self, index):
        # return super().__getitem__(index)
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.insert_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

