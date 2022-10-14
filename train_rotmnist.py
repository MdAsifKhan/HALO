import argparse
from trainer import DynamicalModelTrainer
import torch
import os
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision import datasets, transforms
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import yaml
from dataset import RotatingMNIST2, load_mnist_data
import pdb

import hydra
from omegaconf import DictConfig
from evaluation import evaluation_scores

from einops import rearrange

def run(config, trainer, train_loader, val_loader, results_path, device):
	itern = 1
	if config['trainer']['resume']:
		start = config['trainer']['resume_epoch']
	else:
		start = 0
	for epoch in range(start, config['trainer']['num_epochs']):
		print(f"Training Full Dynamical Model Epoch {epoch}")
		epoch_loss_elbo, epoch_loss_lds, epoch_loss_recon = 0., 0., 0.
		for i, (data_in, labels) in enumerate(train_loader, 0):
			data_in = data_in.to(device[0])
			#labels = labels.to(device[0])
			data_in = rearrange(data_in, 'b t (c w h) -> b t c w h', c=1, w=28, h=28)
			#pdb.set_trace()
			labels = torch.zeros(data_in.shape[0], dtype=torch.int64).to(device[0])
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
			if (epoch + 1) % config['trainer']['anneal_every']:
				if (epoch+1)% 20 == 0 and (epoch+1) >100 and (epoch+1)<=160:
					trainer.config['BetaV'] = 10*trainer.config['BetaV']
					trainer.config['BetaZ'] = 10*trainer.config['BetaZ']
				else:
					trainer.config['BetaV'] = 1 
					trainer.config['BetaZ'] = 1
		if (epoch+1) % config['trainer']['save_every'] == 0:
			trainer.save_model(epoch, sequential=True)
	trainer.writer.close()

def get_config(config):
	with open(config, 'r') as f:
		return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/rmnist.yaml', help='Configuration file')
parser.add_argument('--dataname', type=str, default='rmnist', help='Configuration file')
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


config['trainer']['actions'] = 1 #len(dataset_train.labels_dict)

Xtr, Ytr, Xval, Yval, Xtest, Ytest = load_mnist_data(config['rmnist']['path'])
dataset_train = RotatingMNIST2(Xtr, Ytr)
config['trainer']['nm_seq'] = len(dataset_train.x) 
dataset_val = RotatingMNIST2(Xval, Yval)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config['trainer']['batch_size_train'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=config['trainer']['batch_size_test'], shuffle=True, num_workers=config['num_workers'], drop_last=True)

print(f"Number of training sequence {config['trainer']['nm_seq']}")

config['trainer']['model']['channels'] = 1
config['trainer']['model']['width'] = 28
config['trainer']['model']['height'] = 28
config['trainer']['model']['u_dim'] = 1

labels_dict = {3:0}
config['trainer']['model']['sequential'] = True

trainer_dynamics = DynamicalModelTrainer(config['trainer'], config['data'], labels_dict, results_path, model_path, device)

if config['trainer']['resume']:
	ckpt = f"{model_path}/model_{config['trainer']['resume_epoch']}.pt"
	trainer_dynamics.resume(ckpt, sequential=True)

run(config, trainer_dynamics, train_loader, val_loader, results_path, device)