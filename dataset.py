import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.utils.data as data
import torch
import pdb
from matplotlib import animation
import matplotlib.gridspec as gridspec
from celluloid import Camera
from torchvision import transforms, utils, datasets
import os
from PIL import Image
import glob
from einops import rearrange, repeat

class Sprites(data.Dataset):
	def __init__(self, config, train=True, return_attributes=False):
		self.directions = config['directions']
		self.actions = config['actions']
		self.data_path = config['path']
		self.return_attributes = return_attributes
		if not return_attributes:
			self.data, self.labels_dict, self.labels = self.load(train=train)
		else:
			self.data, self.labels_dict, self.labels, self.attributes_label, self.view_label = self.load(train=train)

		_, self.timesteps, self.rows, self.columns, self.channels = self.data.shape
		self.data = self.data.reshape(-1, self.timesteps, self.channels, self.rows, self.columns)
		if config['tanh']:
			self.data = torch.tanh(self.data)
		
	def load(self, train=True):
		data, labels, labels_all, view_labels_all = [], {}, [], []
		if self.return_attributes:
			attributes_label = []
		for act in range(len(self.actions)):
			label = act
			labels[label] = f"{self.actions[act]}"
			for i in range(len(self.directions)):
				view_label = 3 * act + i
				#labels[label] = f"{self.actions[act]}{self.directions[i]}"
				print(self.actions[act], self.directions[i])
				if train:
					x = np.load(self.data_path + f"{self.actions[act]}_{self.directions[i]}_frames_train.npy")
				else:
					x= np.load(self.data_path + f"{self.actions[act]}_{self.directions[i]}_frames_test.npy")
				data.append(torch.from_numpy(x).float())
				label_d = torch.ones(x.shape[0], dtype=torch.int64)*label
				label_v = torch.ones(x.shape[0], dtype=torch.int64)*view_label
				labels_all.append(label_d)
				view_labels_all.append(label_v)
				if self.return_attributes:
					if train:
						a = np.load(self.data_path + f"{self.actions[act]}_{self.directions[i]}_attributes_train.npy")
					else:
						a = np.load(self.data_path + f"{self.actions[act]}_{self.directions[i]}_attributes_test.npy")					
					attributes_label.append(torch.from_numpy(a))

		data = torch.cat(data, 0)
		labels_all = torch.cat(labels_all, 0)
		view_labels_all = torch.cat(view_labels_all, 0)
		if self.return_attributes:
			attributes_label = torch.cat(attributes_label, 0)
			return data, labels, labels_all.flatten(), attributes_label, view_labels_all.flatten()
		else:
			return data, labels, labels_all.flatten()


	def get_attributes(self, train=True):
		A_data = []
		for act in range(len(self.actions)):
			for i in range(len(self.directions)):
				label = 3 * act + i
				
				if train:
					a = np.load(self.data_path + '%s_%s_attributes_train.npy' % (self.actions[act], self.directions[i]))
				else:
					a = np.load(self.data_path + '%s_%s_attributes_test.npy' % (self.actions[act], self.directions[i]))
				A_data.append(torch.from_numpy(a))
		return A_data


	def __getitem__(self, index):
		if not self.return_attributes:
			return self.data[index], self.labels[index]
		else:
			return self.data[index], self.labels[index], self.attributes_label[index], self.view_label[index]

	def __len__(self):
		return len(self.data)


class MinMaxNormalise(object):
	def __call__(self, im):
		im = 2*(im - im.min())/(im.max()-im.min()) - 1
		return im


class MUG(data.Dataset):
	def __init__(self, config, dataset='TRAIN'):
		self.config = config
		self.transform = transforms.Compose([	
							transforms.ToTensor(),
						])

		video_seq = glob.glob(f"{self.config['path']}/{dataset}/*")

		self.data, self.labels = self.prepare_data(video_seq)			
		self.config['nm_seq'] = len(self.data)
		self.labels_dict = {action:i for i,action in enumerate(self.config['actions'])}
		self.label_to_idx = [self.labels_dict[label] for _,label in self.labels.items()]

	def prepare_data(self, video_seq):
		data, labels = {}, {}
		for idx, seq in enumerate(video_seq):
			action  = seq.split('_')[3]
			seq = glob.glob(f"{seq}/*.jpg")
			data[idx] = seq
			labels[idx] = action
		return data, labels
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		images = []
		for im in self.data[idx]:
			image = Image.open(im)
			image = self.transform(image)
			images.append(image)
		images = torch.stack(images)
		#images = images.permute(0,2,3,1)
		return images, self.label_to_idx[idx]


class RotatingMNIST(data.Dataset):
	def __init__(self, config, 
					train=True):
		self.out_path = config['out_path']
		self.timesteps = config['timesteps']
		self.digits = config['digits']

		if train:
			self.dataset = datasets.MNIST(f"{self.out_path}/MNIST", train=True,
								download=True, transform=transforms.ToTensor())
		else:
			self.dataset = datasets.MNIST(f"{self.out_path}/MNIST", train=False,
								download=True, transform=transforms.ToTensor())


		self.data, self.labels = [], []
		for dg in self.digits:
			data_dg = self.dataset.data[self.dataset.targets==dg]
			self.data.append(data_dg)
			self.labels.append(self.dataset.targets[self.dataset.targets==dg])

		self.data = torch.cat(self.data)
		self.labels = torch.cat(self.labels)
		
		self.data = self.data = (self.data - 125.) / 255

		self.R_theta = lambda theta: torch.Tensor([[torch.cos(theta), -torch.sin(theta), 0],
													[torch.sin(theta), torch.cos(theta), 0]])

		self.theta = torch.linspace(0, 2*np.pi-0.0001, self.timesteps)

		nm_samples = len(self.data)
		self.data_seq = []
		for t in self.theta:
			R_theta_mat = self.R_theta(t)
			R_theta_mat = repeat(R_theta_mat, 'i j -> b i j', b=nm_samples)
			grid = F.affine_grid(R_theta_mat, (nm_samples, 1,28,28))
			data_theta = F.grid_sample(self.data.unsqueeze(1), grid, mode='bilinear', padding_mode='border')
			self.data_seq.append(data_theta.squeeze())

		self.data_seq = torch.stack(self.data_seq).transpose(1,0)
		if not config['tanh']:
			self.data_seq = self.data_seq.mul(0.5).add(0.5)

	def __getitem__(self, i):
		return self.data_seq[i], self.labels[i]		


	def __len__(self):
		return len(self.data_seq)



import scipy.io as sio

def load_mnist_data(path, dt=0.1):
	data = sio.loadmat(f"{path}/rot-mnist-3s.mat")

	Xread = np.squeeze(data['X'])
	Yread = np.squeeze(data['Y'])

	N = np.shape(Xread)[0]
	M = N//10

	tr_idx  = np.arange(0,N-2*M)
	Xtr = Xread[tr_idx,:,:]
	Ytr = dt*np.arange(0,Xtr.shape[1],dtype=np.float32)
	Ytr = np.tile(Ytr,[Xtr.shape[0],1])

	val_idx = np.arange(N-2*M,N-M)
	Xval = Xread[val_idx,:,:]
	Yval  = dt*np.arange(0,Xval.shape[1],dtype=np.float32)
	Yval  = np.tile(Yval,[Xval.shape[0],1])

	test_idx   = np.arange(N-M,N)
	Xtest = Xread[test_idx,:,:]
	Ytest = dt*np.arange(0,Xtest.shape[1],dtype=np.float32)
	Ytest = np.tile(Ytest,[Xtest.shape[0],1])

	return Xtr, Ytr, Xval, Yval, Xtest, Ytest


class RotatingMNIST2(data.Dataset):
	def __init__(self, x, y):
		self.x = x.astype(np.float32)
		self.x = 2*self.x - 1.
		self.y = y

	def plot(self, x, y):
		plt.figure(1,(20,8))
		for j in range(6):
			for i in range(16):
				plt.subplot(7,20,j*20+i+1)
				plt.imshow(np.reshape(x[j,i,:],[28,28]), cmap='gray');
				plt.xticks([]); plt.yticks([])
		plt.show()

	def __getitem__(self, i):
		return self.x[i], self.y[i]

	def __len__(self):
		return len(self.x)