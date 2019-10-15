import torch, os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
from torchvision import datasets, models, transforms
from resModel import ModResnet

params = {
    'batch_size': 8,
    'num_workers': 4,
    'shuffle': True
}



data_transform = transforms.Compose([
				transforms.Resize((512,512)),
				 transforms.ToTensor()])

root_dir = 'path to dataset'

image_datasets = {x:datasets.ImageFolder(os.path.join(root_dir, x), data_transform) for x in ['train', 'valid']}
dataset_size = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], **params) for x in ['train', 'valid']}

# print("IMAGE DATASET TRAIN ::::::",image_da/tasets['train'])
# print("TRAIN DATASET SIZE  ::::::",dataset_size['train'])
# TEMP_ITERATOR = dataloaders['train']

# for x,y in TEMP_ITERATOR:
# 	print(x,y)
# inputs, classes = next(iter(dataloaders['train']))



data_obj = {
	'dataloaders': dataloaders,
	'dataset_size': dataset_size,
	'class_names': class_names
	}

# training 
model = ModResnet(data_obj)
model.train_net()
print('Training Complete & saving model')
model.save_weights('model.pth')