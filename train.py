import argparse
from trainer import DynamicalModelTrainer
import torch
import torch
import pdb
import os
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import datasets, transforms
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import yaml
from celluloid import Camera
from dataset import MUG

import hydra
from omegaconf import DictConfig
from mugdataset import MUGDataset

from video_dataset import VideoDataset, video_transform
import functools


def run(config, trainer, train_loader, results_path, device):
	itern = 1
	if config['trainer']['resume'] and config['trainer']['resume_epoch']>config['trainer']['PretrainEpochs']:
		start = config['trainer']['resume_epoch']
	elif config['trainer']['model']['pretrain']:
		start = config['trainer']['PretrainEpochs']
	else:
		start = 0
	for epoch in range(start, config['trainer']['num_epochs']):
		epoch_loss_lds, epoch_loss_elbo, epoch_loss_recon = 0., 0., 0.
		print(f"Training Full Model Epoch {epoch}")
		for i, (batch) in enumerate(train_loader, 0):
			data_in = batch['images'].permute(0,2,1,3,4).to(device[0])
			labels = batch['categories'].to(device[0])

			labels_onehot = torch.FloatTensor(labels.shape[0], trainer.config['model']['u_dim']).to(device[0])
			labels_onehot.zero_()
			labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

			trainer.update(data_in, labels_onehot)
			trainer.summary(itern)
			itern += 1
			epoch_loss_lds += trainer.lds_loss
			epoch_loss_elbo += trainer.elbo_loss
			epoch_loss_recon += trainer.recon_loss_full

		print(f"Variational Loss for {epoch+1} = {epoch_loss_lds/(i+1)}")
		print(f"ELBO Loss {epoch+1} = {epoch_loss_elbo/(i+1)}")
		print(f"Reconstruction Loss {epoch+1} = {epoch_loss_recon/(i+1)}")
		if config['trainer']['annealkl']:
			if (epoch+1) > 200 and (epoch+1)%20 == 0:
				trainer.config['BetaT'] = 0.5*trainer.config['BetaT']
		if (epoch+1) % config['trainer']['save_every'] == 0:
			trainer.save_model(epoch, sequential=True)
	trainer.writer.close()

def pretrain(config, trainer, train_loader, results_path, device):
	itern = 1
	if config['trainer']['resume']:
		start = config['trainer']['resume_epoch']
	else:
		start = 0
	for epoch in range(start, config['trainer']['PretrainEpochs']):
		print(f"Training Encoder Decoder Epoch {epoch}")
		epoch_loss_elbo, epoch_loss_lds, epoch_loss_recon = 0., 0., 0.
		for i, (batch) in enumerate(train_loader, 0):
			data_in = batch['images'].permute(0,2,1,3,4).to(device[0])
			trainer.pretrain(data_in)
			trainer.pretrain_summary(itern)
			itern += 1
			epoch_loss_lds += trainer.kld_loss_pretrain
			epoch_loss_elbo += trainer.elbo_loss_pretrain
			epoch_loss_recon += trainer.recon_loss_pretrain

		print(f"Variational Loss for {epoch+1} = {epoch_loss_lds/(i+1)}")
		print(f"ELBO Loss {epoch+1} = {epoch_loss_elbo/(i+1)}")
		print(f"Reconstruction Loss {epoch+1} = {epoch_loss_recon/(i+1)}")

		if (epoch+1) % config['trainer']['save_every'] == 0:
			trainer.save_model(epoch, sequential=False)
	trainer.writer.close()

def get_config(config):
	with open(config, 'r') as f:
		return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/sprites.yaml', help='Configuration file')
parser.add_argument('--dataname', type=str, default='MUG', help='Configuration file')
opt = parser.parse_args()


config = get_config(opt.config)

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if not config['cuda']:
	device = ['cpu']
else:
	device = config['gpu_ids']



results_path = f"{config['output_results']}_{config['data']}_TemporalLoss_{config['trainer']['temporalLoss']}_TempLossBeta_{config['trainer']['BetaT']}"\
		f"_condnSonU_{config['trainer']['model']['condnSonU']}_maskV_{config['trainer']['model']['projection']}"\
		f"_dynamics_{config['trainer']['model']['dynamics']}_qv_x_{config['trainer']['model']['qv_x']}"\
		f"_betaV_{config['trainer']['BetaV']}_useZ_{config['trainer']['model']['useZ']}_Zlstm_{config['trainer']['model']['uselstmZ']}"\
		f"_reconVloss_{config['trainer']['lossVrecon']}_reconxloss_{config['trainer']['reconloss']}"

model_path = f"{config['output_model']}_{config['data']}_TemporalLoss_{config['trainer']['temporalLoss']}_TempLossBeta_{config['trainer']['BetaT']}"\
		f"_condnsonU_{config['trainer']['model']['condnSonU']}_maskV_{config['trainer']['model']['projection']}"\
		f"_dynamics_{config['trainer']['model']['dynamics']}_qv_x_{config['trainer']['model']['qv_x']}"\
		f"_betaV_{config['trainer']['BetaV']}_useZ_{config['trainer']['model']['useZ']}_Zlstm_{config['trainer']['model']['uselstmZ']}"\
		f"_reconVloss_{config['trainer']['lossVrecon']}_reconxloss_{config['trainer']['reconloss']}"


if not os.path.exists(results_path):
	os.makedirs(results_path)
if not os.path.exists(model_path):
	os.makedirs(model_path)

if config['trainer']['reconloss'] in ['l2', 'l1']:
	image_transforms = transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					])
elif config['trainer']['reconloss'] == 'bce':
	image_transforms = transforms.Compose([
						transforms.ToTensor(),
					])
else:
	assert 0,f"Not implemented {config['trainer']['reconloss']}"


if dataname == 'mug':
	dataset_train = MUGDataset(config['mug'], 'Train')
	video_transforms = functools.partial(video_transform, image_transform=image_transforms)

	dataset_ = VideoDataset(dataset_train, 8, 2, transform=video_transforms)

	train_loader = torch.utils.data.DataLoader(dataset_, batch_size=config['trainer']['batch_size_train'], 
								shuffle=True, num_workers=4, drop_last=True)

	config['mug']['timesteps'] = dataset_train.config['timesteps']
	config['trainer']['nm_seq'] = len(dataset_train.lines)
	print(f"Number of training sequence {config['trainer']['nm_seq']}")

	config['trainer']['model']['channels'] = dataset_train.config['channels']
	config['trainer']['model']['width'] = dataset_train.config['rows']
	config['trainer']['model']['height'] = dataset_train.config['columns']
	config['trainer']['model']['u_dim'] = len(dataset_train.config['actions'])

elif dataname == 'sprites':
	dataset_train = Sprites(config['sprites'], train=True)
	config['sprites']['timesteps'] = dataset_train.timesteps
	config['trainer']['actions'] = len(dataset_train.labels_dict)
	config['trainer']['nm_seq'] = len(dataset_train.labels)

	train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config['trainer']['batch_size_train'], shuffle=True, num_workers=config['num_workers'], drop_last=True)

	print(f"Number of training sequence {config['trainer']['nm_seq']}")

	config['trainer']['model']['channels'] = dataset_train.channels
	config['trainer']['model']['width'] = dataset_train.rows
	config['trainer']['model']['height'] = dataset_train.columns
	config['trainer']['model']['u_dim'] = len(dataset_train.labels_dict)

else:
	assert 0,f"{dataname} Not Implemented"


if config['trainer']['model']['pretrain']:
	config['trainer']['model']['sequential'] = False
	trainer = DynamicalModelTrainer(config['trainer'], config['data'], dataset_train.labels_dict, results_path, model_path, device)
	if config['trainer']['resume']:
		if config['trainer']['resume_epoch']<config['trainer']['PretrainEpochs']:
			ckpt = f"{model_path}/model_{config['trainer']['resume_epoch']}.pt" 
			trainer.resume(ckpt, sequential=False)
			pretrain(config, trainer, train_loader, results_path, device)
		else:
			del trainer
			config['trainer']['model']['sequential'] = True
			trainer_dynamics = DynamicalModelTrainer(config['trainer'], config['data'], dataset_train.labels_dict, results_path, model_path, device)
			ckpt = f"{model_path}/model_{config['trainer']['resume_epoch']}.pt" 
			trainer_dynamics.resume(ckpt, sequential=True)
			run(config, trainer_dynamics, train_loader, results_path, device)
	else:
		pretrain(config, trainer, train_loader, results_path, device) 
	
	del trainer
	config['trainer']['model']['sequential'] = True
	trainer_dynamics = DynamicalModelTrainer(config['trainer'], config['data'], dataset_train.labels_dict, results_path, model_path, device)

	ckpt = f"{model_path}/model_{config['trainer']['PretrainEpochs']}.pt"
	trainer_dynamics.resume(ckpt, sequential=False)
	run(config, trainer_dynamics, train_loader, results_path, device)

else:
	config['trainer']['model']['sequential'] = True
	trainer_dynamics = DynamicalModelTrainer(config['trainer'], config['data'], dataset_train.labels_dict, results_path, model_path, device)
	
	if config['trainer']['resume']:
		ckpt = f"{model_path}/model_{config['trainer']['resume_epoch']}.pt"
		trainer_dynamics.resume(ckpt, sequential=True)

	run(config, trainer_dynamics, train_loader, results_path, device)
