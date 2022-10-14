import argparse
from trainer import DynamicalModelTrainer
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
from utils import save_latent_z3D, latent_scatter_3d, latent_scatter_2d, atanh
from dataset import MUG

from torchvision import transforms, datasets
#from sklearn.decomposition import PCA
from evaluation import evaluation_scores
import functools
from mugdataset import MUGDataset
from video_dataset import VideoDataset
from utils import save_seq_img, video_transform

def compute_energy(config, trainer, test_loader, results_path, device):
	time = 32
	with torch.no_grad():
		print("Testing")
		coords_K = {i: [] for i in range(trainer.vae.lds.nm_operators)}
		for j, (data) in enumerate(test_loader, 0):
			print(f"Batch {j}")
			# Reconstruct Sequences
			data_test = data['images'].permute(0,2,1,3,4).to(device[0])
			labels = data['categories'].to(device[0])

			batch_size, _, channels, rows, columns = data_test.shape
			labels_onehot = torch.FloatTensor(labels.shape[0], trainer.config['model']['u_dim']).to(device[0])
			labels_onehot.zero_()
			labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

			x_gen, x_t, energy, coords, KE, PE, MIX = trainer.hamiltonian_energy(data_test, labels_onehot, time)
		
			for i in range(batch_size):
				for k in range(config['model']['nm_operators']):
					plt.plot(range(time), energy[i][k,:].cpu().numpy(), label=f"E-{k+1}")
					plt.plot(range(time), KE[i][k,:].cpu().numpy(), label=f"KE-{k+1}")
					plt.plot(range(time), PE[i][k,:].cpu().numpy(), label=f"PE-{k+1}")
					plt.plot(range(time), MIX[i][k,:].cpu().numpy(), label=f"NonSep-{k+1}")
					plt.legend(loc='upper right')
					plt.xlabel('time ->')
					plt.ylabel('Energy ')
					plt.savefig(f"{results_path}/energy_sample_{i}_batch_{j}_operator_{k+1}.png")
					plt.cla()
					plt.clf()
			for i, category in enumerate(data['categories']):
				coords_K[category.item()].append(coords[i,:,category.item(),:])

		grid_x = torch.linspace(-1, 1, 100)
		grid_y = torch.linspace(-1, 1, 100)
		data = torch.FloatTensor(100, 100, 2)
		for i, x in enumerate(grid_x):
			for j, y in enumerate(grid_y):
				data[i, j, 0] = x
				data[i, j, 1] = y
		data = data.view(-1, 2).numpy()
		energy_grid = {i: None for i in range(trainer.vae.lds.nm_operators)}
		pca_K = {i:None for i in range(trainer.vae.lds.nm_operators)}
		for k in range(len(coords_K)):
			coords_c = torch.cat(coords_K[k], 0).cpu().numpy()
			from sklearn.decomposition import PCA
			pca = PCA(n_components=2)
			pca.fit(coords_c)
			pca_K[k] = pca.transform(coords_c)
			data_project = pca.inverse_transform(data)
			data_project = torch.from_numpy(data_project).to(x_t.device)
			M = (trainer.vae.lds.H[k] + trainer.vae.lds.H[k].transpose(1,0))*0.5
			energy_j = torch.einsum('b d, d k, b k -> b', data_project, M, data_project)
			energy_grid[k] = energy_j

		fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
		m = 0
		for k in range(2):
			for l in range(3):
				axs[k, l].scatter(data[:,0], data[:,1], c=energy_grid[m].cpu().numpy())
				m += 1
				axs[k, l].set_xlabel('S 1')
				axs[k, l].set_xlabel('S 2')
		fig.savefig(f"{results_path}/rand_energy.png")
		plt.cla()
		plt.clf()
		fig, axs = plt.subplots(2,3, sharex=True, sharey=True)
		m = 0
		for k in range(2):
			for l in range(2):
				axs[k,l].scatter(pca_K[m][:,0], pca_K[m][:,1], label= f"H-{m+1}")
				m += 1
				axs[k, l].set_xlabel('PC 1')
				axs[k, l].set_ylabel('PC 2')
		fig.savefig(f"{results_path}/pca_phase_space.png")	
	trainer.writer.close()


def get_config(config):
	with open(config, 'r') as f:
		return yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/sprites.yaml', help='Configuration file')
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

results_path = f"{config['output_results']}_{config['data']}_pretrain{config['trainer']['model']['pretrain']}"\
		f"TemporalLoss_{config['trainer']['temporalLoss']}_TempLossBeta_{config['trainer']['BetaT']}"\
		f"_splitV_{config['trainer']['model']['condnSonU']}_maskV_{config['trainer']['model']['projection']}"\
		f"_dynamics_{config['trainer']['model']['dynamics']}_qv_x_{config['trainer']['model']['qv_x']}"\
		f"_betaV_{config['trainer']['BetaV']}_contentlstm_{config['trainer']['model']['uselstmZ']}"\
		f"_reconVloss_{config['trainer']['lossVrecon']}_reconloss_{config['trainer']['reconloss']}"

model_path = f"{config['output_model']}_{config['data']}_pretrain{config['trainer']['model']['pretrain']}"\
		f"TemporalLoss_{config['trainer']['temporalLoss']}_TempLossBeta_{config['trainer']['BetaT']}"\
		f"_splitV_{config['trainer']['model']['condnSonU']}_maskV_{config['trainer']['model']['projection']}"\
		f"_dynamics_{config['trainer']['model']['dynamics']}_qv_x_{config['trainer']['model']['qv_x']}"\
		f"_betaV_{config['trainer']['BetaV']}_contentlstm_{config['trainer']['model']['uselstmZ']}"\
		f"_reconVloss_{config['trainer']['lossVrecon']}_reconloss_{config['trainer']['reconloss']}"


ckpt = f"{model_path}/model_{config['test_epoch']}.pt"
state = torch.load(ckpt, map_location=torch.device('cpu'))
if not os.path.exists(results_path):
	os.makedirs(results_path)

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


dataset = MUGDataset(config['mug'], 'Test')
print(f"Number of test sequence {len(dataset.lines)}")
config['mug']['timesteps'] = dataset.config['timesteps']
config['trainer']['actions'] = len(dataset.labels_dict)

video_transforms = functools.partial(video_transform, image_transform=image_transforms)

test_loader = torch.utils.data.DataLoader(VideoDataset(dataset, 8, 2, transform=video_transforms), 
												batch_size=config['trainer']['batch_size_test'], 
												shuffle=True, num_workers=4, drop_last=True)

config['trainer']['model']['channels'] = dataset.config['channels']
config['trainer']['model']['width'] = dataset.config['rows']
config['trainer']['model']['height'] = dataset.config['columns']
config['trainer']['model']['u_dim'] = len(config['mug']['actions'])

config['trainer']['model']['sequential'] = True
trainer = DynamicalModelTrainer(config['trainer'], config['data'], state['actions'], results_path, model_path, device)

trainer.vae.encoder.load_state_dict(state['encoder'])
trainer.vae.decoder.load_state_dict(state['decoder'])
if config['trainer']['model']['sequential']:
	trainer.vae.lds.load_state_dict(state['lds'])

compute_energy(config['trainer'], trainer, test_loader, results_path, device)

