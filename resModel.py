import os, time, copy
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import torch


class ModResnet(object):

	def __init__(self, dataobj):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.dataobj = dataobj
		self.dataloaders = dataobj['dataloaders']
		self.dataset_size = dataobj['dataset_size']
		self.class_names = dataobj['class_names']
		self.acc_plot = {'train': [], 'valid': []}
		self.loss_plot = {'train': [], 'valid': []}

		self.model = models.resnet18(pretrained=True)
		num_filters = self.model.fc.in_features
		self.model.fc = nn.Linear(num_filters, len(self.dataobj['class_names']))
		self.model = self.model.to(self.device)
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
		self.schedular = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

	def __call__(self, inp_tensor):
		print('#'*20)
		print('__call__ MODE')
		self.model.eval()
		return self.model(inp_tensor)

	def train_net(self):
		since = time.time()
		best_model_wts = copy.deepcopy(self.model.state_dict())
		best_acc = 0.0
		num_epochs = 20
		for epoch in range(num_epochs):
			print('Epoch{}/{}'.format(epoch+1, num_epochs))
			print('*'*10)

			for phase in ['train', 'valid']:
				if phase == 'train':
					self.schedular.step() # change the learning rate
					self.model.train()
				else:
					self.model.eval()

				running_loss = 0.0
				running_corrects = 0
				for inputs, labels in self.dataloaders[phase]:
					inputs = inputs.to(self.device)
					labels = labels.to(self.device)

					self.optimizer.zero_grad()

					with torch.set_grad_enabled(phase == 'train'):
						outputs = self.model(inputs)
						_, preds = torch.max(outputs, 1)
						loss = self.criterion(outputs, labels)
						if phase =='train':
							loss.backward()
							self.optimizer.step()

					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)

				epoch_loss = running_loss / self.dataset_size[phase]
				epoch_acc  = running_corrects.double() / self.dataset_size[phase] 
				self.acc_plot[phase].append(epoch_acc)
				self.loss_plot[phase].append(epoch_loss)

				print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

				if phase == 'valid' and epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(self.model.state_dict())
			print()
		print('Best Val Accuracy :', best_acc)

		self.model.load_state_dict(best_model_wts)

	def save_weights(self, filename):
		torch.save(self.model, filename)
		print('model saved in :', filename)
