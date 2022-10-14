import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import numpy as np
import pdb


from base import DynamicalModel
from model import VAE, SubspaceVAE, Discriminator_V
from einops import rearrange, repeat

class DynamicalModelTrainer(nn.Module):
	def __init__(self, config, 
				data, 
				labels_dict, 
				results_path, 
				model_path, 
				device):
		super(DynamicalModelTrainer, self).__init__()
		self.config = config
		self.results_path = results_path
		self.model_path = model_path
		self.device = device
		self.labels_dict = labels_dict

		self.writer = SummaryWriter('{}/summary'.format(self.results_path))		

		self.vae = SubspaceVAE(self.config['model'], data, device_ids=device)


		self.optim = torch.optim.Adam(list(self.vae.encoder.parameters()) + list(self.vae.decoder.parameters()), 
							lr=self.config['optim']['lr'], 
							betas=(self.config['optim']['beta1'], 
							self.config['optim']['beta2']))

		if self.config['model']['sequential'] and self.config['model']['dynamics'] != 'Fourier':
			self.optimlds = torch.optim.Adam(self.vae.lds.parameters(), 
								lr=self.config['optim']['lr'], 
								betas=(self.config['optim']['beta1'], 
								self.config['optim']['beta2']))

	def logPx(self, x, x_recon):
		x = rearrange(x, 'b t c w h -> (b t) (c w h)')
		x_recon = rearrange(x_recon, 'b t c w h -> (b t) (c w h)')
		if self.config['reconloss'] == 'bce':
			return F.binary_cross_entropy(x_recon, x, reduction='sum')
		elif self.config['reconloss'] == 'l2':
			return F.mse_loss(x, x_recon, reduction='sum')
		elif self.config['reconloss'] == 'l1':
			return F.l1_loss(x, x_recon, reduction='sum')
		else:
			assert 0, f"Not impelemented {self.config['reconloss']}"

	def logPz(self, z, z_recon):
		z = rearrange(z, 'b t z -> (b t) z')
		z_recon = rearrange(z_recon, 'b t z -> (b t) z')
		if self.config['reconloss'] == 'l2':
			return F.mse_loss(z, z_recon, reduction='sum')
		elif self.config['reconloss'] == 'l1':
			return F.l1_loss(z, z_recon, reduction='sum')
		else:
			assert 0,f"Not impelemented {self.config['reconloss']}"

	def kl_subspace_loss(self, mu, logsigma, labels):
		labels = labels.argmax(axis=1)
		idx = range(len(labels))
		kld = -0.5 * torch.sum(1 + logsigma[idx,:,labels,:] - mu[idx,:,labels,:].pow(2) - logsigma[idx,:,labels,:].exp())
		return kld


	def pretrain(self, x_t):
		self.optim.zero_grad()
		batch, time, _, _, _ = x_t.shape
		z, mu, logsigma = self.vae.pretrainEncode(x_t)
	
		x_t_recon = self.vae.decode(z.to(self.device[1])).to(self.device[0])

		self.recon_loss_pretrain = self.logPx(x_t, x_t_recon)

		mu = rearrange(mu, 'b t z -> (b t) z')
		logsigma = rearrange(logsigma, 'b t z -> (b t) z')

		self.kld_loss_pretrain = self.vae.vae_loss(mu, logsigma)
		self.kld_loss_pretrain = self.kld_loss_pretrain/(batch*time)
		self.recon_loss_pretrain = self.recon_loss_pretrain/(batch*time)
		self.elbo_loss_pretrain = (self.kld_loss_pretrain*self.config['BetaZ'] + self.config['recon_term']*self.recon_loss_pretrain)
		self.elbo_loss_pretrain.backward()
		self.optim.step()

	def update(self, x_t, u_t):
		self.optim.zero_grad()
		batch, time, _, _, _ = x_t.shape
		if self.config['model']['dynamics'] != 'Fourier':
			self.optimlds.zero_grad()
		
		z, mu_z, logsigma_z, q_t, mu_qt, logsigma_qt, p_t, mu_pt, logsigma_pt = self.vae.encode(x_t, u_t)
		subspace_dim = q_t.shape[-1]
		if self.config['model']['dynamics'] in ['Fourier', 'RNN', 'Linear']:
			s_t = q_t
			s_t_gen, startidx = self.vae.dynamics(s_t, u_t, time)
			subspace_dim = s_t_gen.shape[-1]

		else:
			s_t = torch.cat([q_t, p_t], 3)
			s_t_gen, startidx = self.vae.dynamics(s_t, u_t, time)

		if self.config['model']['useZ']:
			#Concatenate Content and Motion Variables
			z = repeat(z, 'b z -> b t z', t=time)
			zs_t_gen = torch.cat([z, rearrange(s_t_gen[:,:,:,:subspace_dim], 'b t n d -> b t (n d)')], dim=2)
			kld_z = self.vae.vae_loss(mu_z, logsigma_z)
		else:
			zs_t_gen = rearrange(s_t_gen[:,:,:,:subspace_dim], 'b t n d -> b t (n d)')
			kld_z = 0.

		x_t_recon = self.vae.decode(zs_t_gen.to(self.device[1])).to(self.device[0])


		if self.config['model']['dynamics'] == 'Fourier':
			kld_st = self.vae.vae_loss(mu_qt, logsigma_qt)
			s_loss, x_recon_f_loss, temploss = 0., 0., 0.

		elif self.config['model']['dynamics'] in ['RNN', 'Linear']:
			if self.config['model']['condnSonU']:
				kld_st = self.kl_subspace_loss(mu_qt, logsigma_qt, u_t)
			else:
				kld_st = self.vae.vae_loss(rearrange(mu_qt, 'b t n d -> (b t) (n d)'), rearrange(logsigma_qt, 'b t n d -> (b t) (n d)')) 
			#s_loss, x_recon_f_loss, temploss = 0., 0., 0.
		else:
			if self.config['model']['condnSonU']:
				kld_st = self.kl_subspace_loss(mu_qt, logsigma_qt, u_t)
				kld_st += self.kl_subspace_loss(mu_pt, logsigma_pt, u_t)
			else:
				kld_st = self.vae.vae_loss(rearrange(mu_qt, 'b t n d -> (b t) (n d)'), rearrange(logsigma_qt, 'b t n d -> (b t) (n d)'))
				kld_st += self.vae.vae_loss(rearrange(mu_pt, 'b t n d -> (b t) (n d)'), rearrange(logsigma_pt, 'b t n d -> (b t) (n d)'))

		if self.config['lossVrecon']:
			if self.config['model']['dynamics'] == 'Fourier':
				s_loss, x_recon_f, x_recon_f_loss = 0., 0., 0.
			else:
				s_loss = self.logPz(rearrange(s_t_gen, 'b t n d -> b t (n d)'), rearrange(s_t, 'b t n d -> b t (n d)'))/batch
				if self.config['model']['useZ']:
					z_s_t = torch.cat([z, rearrange(s_t[:,:,:,:subspace_dim], 'b t n d -> b t (n d)')], dim=2)
				else:
					z_s_t = rearrange(s_t[:,:,:,:subspace_dim], 'b t n d -> b t (n d)')
	
				x_recon_f = self.vae.decode(z_s_t.to(self.device[1])).to(self.device[0])
				x_recon_f_loss = self.logPx(x_t, x_recon_f)/batch


		if self.config['temporalLoss']:
			x_orig_diff = x_t[:,1:,:,:,:]  - x_t[:,:-1,:,:,:] 
			x_gen_diff = x_t_recon[:,1:,:,:,:]  - x_t_recon[:,:-1,:,:,:]
			temploss = self.logPx(x_orig_diff, x_gen_diff)/batch
		else:
			temploss = 0.

		full_kl = (self.config['BetaZ']*kld_z + self.config['BetaV']*kld_st)/batch
		recon_loss_full = self.config['recon_term'] * (self.logPx(x_t, x_t_recon)/batch + x_recon_f_loss) + self.config['BetaV1']*s_loss

		elbo_loss = recon_loss_full + full_kl + self.config['BetaT']*temploss

		elbo_loss.backward()

		self.temploss = temploss
		self.lds_loss = full_kl.item()
		self.recon_loss_full = recon_loss_full.item()
		self.elbo_loss = elbo_loss.item()

		self.optim.step()
		if self.config['model']['dynamics'] != 'Fourier':
			self.optimlds.step()


	def valloss(self, x_t, u_t):
		self.vae.eval()

		z, mu_z, logsigma_z, q_t, mu_qt, logsigma_qt, p_t, mu_pt, logsigma_pt = self.vae.encode(x_t, u_t)
		subspace_dim = q_t.shape[-1]
		if self.config['model']['dynamics'] in ['Fourier', 'Linear', 'RNN']:
			s_t_gen, startidx = self.vae.dynamics(mu_qt, u_t, time, test=True)
			subspace_dim = s_t_gen.shape[-1]
		else:
			s_t = torch.cat([q_t, p_t], 3)
			
		batch, time, nm_subspaces, _ = s_t.shape

		if self.config['model']['useZ']:
			z = repeat(z, 'b z -> b t z', t=time)
			zs_t_gen = torch.cat([z, rearrange(s_t_gen[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2)
			kld_z = self.vae.vae_loss(mu_z, logsigma_z)    
		else:
			zs_t_gen = rearrange(s_t_gen[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')
			kld_z = 0.

		x_t_recon = self.vae.decode(zs_t_gen.to(self.device[1])).to(self.device[0])
	
		if self.config['model']['dynamics'] == 'Fourier':
			kld_st = self.vae.vae_loss(mu_qt, logsigma_qt)
			s_loss, x_recon_f_loss, temploss = 0., 0., 0.

		elif self.config['model']['dynamics'] in ['RNN', 'Linear']:	
			if self.config['model']['condnSonU']:
				kld_st = self.kl_subspace_loss(mu_qt, logsigma_qt, u_t)
			else:
				kld_st = self.vae.vae_loss(rearrange(mu_qt, 'b t n v -> (b t) (n v)'), rearrange(logsigma_qt, 'b t n v -> (b t) (n v)')) 
			#s_loss, x_recon_f_loss, temploss = 0., 0., 0.
		else:
			if self.config['model']['condnSonU']:
				kld_st = self.kl_subspace_loss(mu_qt, logsigma_qt, u_t) 
				kld_st += self.kl_subspace_loss(mu_pt, logsigma_pt, u_t)
			else:
				kld_st = self.vae.vae_loss(mu_qt.squeeze(), logsigma_qt.squeeze())
				kld_st += self.vae.vae_loss(mu_pt.squeeze(), logsigma_pt.squeeze())


		if self.config['lossVrecon']:
			if self.config['model']['dynamics'] == 'Fourier':
				s_loss, x_recon_f, x_recon_f_loss = 0., 0., 0.
			else:
				s_loss = self.logPz(rearrange(s_t_gen, 'b t n v -> b t (n v)'), rearrange(s_t, 'b t n v -> b t (n v)'))/batch
				if self.config['model']['useZ']:
					z_s_t = torch.cat([z, rearrange(s_t[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2)
				else:
					z_s_t = rearrange(s_t[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')

				x_recon_f = self.vae.decode(z_s_t.to(self.device[1])).to(self.device[0])
				x_recon_f_loss = self.logPx(x_t, x_recon_f)/batch


		if self.config['temporalLoss']:
			x_orig_diff = x_t[:,1:,:,:,:]  - x_t[:,:-1,:,:,:] 
			x_gen_diff = x_t_recon[:,1:,:,:,:]  - x_t_recon[:,:-1,:,:,:]
			temploss = self.logPx(x_orig_diff, x_gen_diff)
			temploss = temploss/batch
		else:
			temploss = 0.


		full_kl = (self.config['BetaZ']*kld_z + self.config['BetaV']*kld_st)/batch
		recon_s = (self.config['BetaV1']*s_loss)/batch
		recon_loss_full = self.config['recon_term'] * ((self.logPx(x_t, x_t_recon)/batch + x_recon_f_loss) + recon_s)


		elbo_loss = recon_loss_full + full_kl + self.config['BetaT']*temploss

		self.temploss_val = temploss
		self.recon_loss_val = recon_loss_full
		self.lds_loss_val = lds_loss_val
		self.elbo_loss_val = elbo_loss_val

		self.vae.train()


	def save_model(self, epoch, sequential=False):
		model = os.path.join(self.model_path, 'model_{}.pt'.format(epoch+1))
		if not sequential:
			torch.save({'encoder':self.vae.encoder.state_dict(), 'decoder':self.vae.decoder.state_dict(), 
					'actions':self.labels_dict, 'optim':self.optim.state_dict()}, model)
		else:
			if self.config['model']['dynamics'] != 'Fourier':
				torch.save({'encoder':self.vae.encoder.state_dict(), 'decoder':self.vae.decoder.state_dict(),
						'lds':self.vae.lds.state_dict(), 'actions':self.labels_dict, 
						'optim':self.optim.state_dict(), 'optimlds':self.optimlds.state_dict()}, model)
			else:
				torch.save({'encoder':self.vae.encoder.state_dict(), 'decoder':self.vae.decoder.state_dict(),
						'actions':self.labels_dict, 'optim':self.optim.state_dict()}, model)


	def resume(self, checkpoint, sequential=False, device=torch.device('cpu')):
		state = torch.load(checkpoint, map_location=device)
		self.vae.encoder.load_state_dict(state['encoder'])
		self.vae.decoder.load_state_dict(state['decoder'])
		self.optim.load_state_dict(state['optim'])
		if sequential and self.config['model']['dynamics'] != 'Fourier':
			self.vae.lds.load_state_dict(state['lds'])
			self.optimlds.load_state_dict(state['optimlds'])

	def summary(self, i):
		self.writer.add_scalar('Iter_Reconstruction_Loss', self.recon_loss_full, i)
		self.writer.add_scalar('Iter_Variational_Loss', self.lds_loss, i)
		self.writer.add_scalar('Iter_VAE_Loss', self.elbo_loss, i)
		self.writer.add_scalar('Iter_Temporal_Loss', self.temploss, i)

	def pretrain_summary(self, i):
		self.writer.add_scalar('Pretrain_Iter_Reconstruction_Loss', self.recon_loss_pretrain, i)
		self.writer.add_scalar('Pretrain_Iter_Variational_Loss', self.kld_loss_pretrain, i)
		self.writer.add_scalar('Pretrain_Iter_VAE_Loss', self.elbo_loss_pretrain, i)

	def test_summary(self, i):
		self.writer.add_scalar('Iter_Test_Reconstruction_Loss', self.recon_loss_val, i)
		self.writer.add_scalar('Iter_Test_Variational_Loss', self.lds_loss_val, i)
		self.writer.add_scalar('Iter_Test_Total_VAE_Loss', self.elbo_loss_val, i)
		self.writer.add_scalar('Iter_Test_Temporal_Loss', self.temploss_val, i)


	def reconstruct(self, x_t, u_t, time=None):
		self.vae.eval()
		if not time:
			time = x_t.shape[1]

		z, mu_z, logsigma_z, q_t, mu_qt, logsigma_qt, p_t, mu_pt, logsigma_pt = self.vae.encode(x_t, u_t)
		subspace_dim = q_t.shape[-1]
		if self.config['model']['dynamics'] == 'Fourier':
			s_t = q_t
			s_t, startidx = self.vae.dynamics(s_t, u_t, time)
			subspace_dim = s_t.shape[-1]
		elif self.config['model']['dynamics'] in ['RNN', 'Linear']:
			s_t = mu_qt
		else:
			s_t = torch.cat([mu_qt, mu_pt], 3)

		if self.config['model']['useZ']:
			#Concatenate Content and Motion Variables
			mu_z = repeat(mu_z, 'b z -> b t z', t=time)
			z_s_t_gen = torch.cat([mu_z, rearrange(s_t[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2) 
		else:
			z_s_t_gen = s_t[:,:,:,:subspace_dim]

		x_recon = self.vae.decode(z_s_t_gen.to(self.device[1])).to(self.device[0])

		self.vae.train()
		return x_recon, x_t

	def generate(self, x_t, u_t, time=None):
		self.vae.eval()
		if not time:
			time = x_t.shape[1]

		z, mu_z, logsigma_z, q_t, mu_qt, logsigma_qt, p_t, mu_pt, logsigma_pt = self.vae.encode(x_t, u_t)
		subspace_dim = q_t.shape[-1]
		if self.config['model']['dynamics'] == 'Fourier':
			s_t = q_t
			s_t, startidx = self.vae.dynamics(s_t, u_t, time)
			subspace_dim = s_t.shape[-1]
		elif self.config['model']['dynamics'] in ['RNN', 'Linear']:
			s_t = q_t
			subspace_dim = q_t.shape[-1]
		else:
			s_t = torch.cat([mu_qt, mu_pt], 3)
			
		if self.config['model']['useZ']:
			mu_z = repeat(mu_z, 'b z -> b t z', t=time)
			z_s_t_recon = torch.cat([mu_z, rearrange(s_t[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2)
		else:
			z_s_t_recon = rearrange(s_t[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')

		x_recon = self.vae.decode(z_s_t_recon.to(self.device[1])).to(self.device[0])


		if self.config['model']['dynamics'] == 'Fourier':
			s_t_gen = s_t
		elif self.config['model']['dynamics'] in ['RNN', 'Linear']:
			startidx = 0
			s_f = s_t[:,0,:,:]
			s_t_gen = self.vae.lds(u_t.to(self.device[2]), s_f.to(self.device[2]), time, startidx, test=True).to(self.device[0])
		elif self.config['model']['dynamics'] in ['Hamiltonian', 'skewHamiltonian']:
			startidx = 3
			s_f = s_t[:,startidx,:,:]
			s_t_gen = self.vae.lds(u_t.to(self.device[2]), s_f.to(self.device[2]), time, startidx).to(self.device[0])
		
		if self.config['model']['useZ']:
			#Concatenate Content and Motion Variables
			z_s_t_gen = torch.cat([mu_z, rearrange(s_t_gen[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2)
		else:
			z_s_t_gen = rearrange(s_t_gen[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')

		x_gen = self.vae.decode(z_s_t_gen.to(self.device[1])).to(self.device[0])

		x_gen = torch.cat([x_gen[:,startidx:,:,:,:], x_gen[:,:startidx,:,:,:]], 1)
		x_t = torch.cat([x_t[:,startidx:,:,:,:], x_t[:,:startidx,:,:,:]], 1)
		x_recon = torch.cat([x_recon[:,startidx:,:,:,:], x_recon[:,:startidx,:,:,:]], 1)
		self.vae.train()
		return x_gen, x_recon, x_t

	def hamiltonian_energy(self, x_t, u_t, time=8):
		self.vae.eval()

		z, mu_z, logsigma_z, q_t, mu_qt, logsigma_qt, p_t, mu_pt, logsigma_pt = self.vae.encode(x_t, u_t)

		if self.config['model']['dynamics'] not in ['Hamiltonian', 'SkewHamiltonian']:
			assert 0,f"THe dynamical model {self.config['model']['dynamics']} is not Hamiltonian"

		coords_in = torch.cat([mu_qt, mu_pt], 3)
		frameidx_t = 3	
		image = x_t[:,frameidx_t,:,:,:].unsqueeze(1)
		image = repeat(image, 'b 1 c w h -> b t c w h', t=time)
		
		rand_label = torch.linspace(0, self.config['actions'] - 1, self.config['actions'], dtype=int).to(x_t.device)
		u_t = torch.FloatTensor(len(rand_label), self.config['model']['u_dim']).to(x_t.device)
		u_t.zero_()
		u_t.scatter_(1, rand_label.unsqueeze(1), 1)

		energy = {i: None for i in range(image.shape[0])}
		KE = {i: None for i in range(image.shape[0])}
		PE = {i: None for i in range(image.shape[0])}
		NSEP = {i: None for i in range(image.shape[0])}
		x_recon = []	
		for i, x_i in enumerate(image):
			x_i = repeat(x_i, 't c w h -> n t c w h', n=len(rand_label))
			_, mu_z_i, _, q_t_i, mu_qt_i, _, _, mu_pt_i, _ = self.vae.encode(x_i, u_t)
			s_t = torch.cat([mu_qt_i, mu_pt_i], 3)
			coords = self.vae.lds(u_t.to(self.device[2]), s_t[:,frameidx_t,:,:].squeeze(1).to(self.device[2]), time, frameidx_t).to(self.device[0]) 
			M = (self.vae.lds.H + self.vae.lds.H.transpose(2,1))*0.5
			M = repeat(M,'n i j -> k n i j', k=len(rand_label))

			s_d = M.shape[-1]
			A, B  = M[:,:,:int(s_d/2),:int(s_d/2)], M[:,:,:int(s_d/2),int(s_d/2):]
			C, D = M[:,:,int(s_d/2):,:int(s_d/2)], M[:,:,int(s_d/2):,int(s_d/2):]

			energy_j = 0.5*torch.einsum('n t m d, n m d k, n t m k -> n t', coords, M, coords)
			PE[i] = 0.5*torch.einsum('n t m d, n m d k, n t m k -> n t', coords[:,:,:,:int(s_d/2)], A, coords[:,:,:,:int(s_d/2)])
			KE[i] = 0.5*torch.einsum('n t m d, n m d k, n t m k -> n t', coords[:,:,:,int(s_d/2):], D, coords[:,:,:,int(s_d/2):])
			NSEP[i] = 0.5*torch.einsum('n t m d, n m d k, n t m k -> n t', coords[:,:,:,:int(s_d/2)], B, coords[:,:,:,int(s_d/2):])
			NSEP[i] += 0.5*torch.einsum('n t m d, n m d k, n t m k -> n t', coords[:,:,:,int(s_d/2):], C, coords[:,:,:,:int(s_d/2)])
			energy[i] = energy_j

			mu_z_i = repeat(mu_z_i, 'b z -> b t z', t=time)
			z_q_gen_i = torch.cat([mu_z_i, rearrange(coords[:,:,:,:int(s_d/2)], 'b t n v -> b t (n v)')], dim=2)
			
			x_reconi = self.vae.decode(z_q_gen_i)
			x_reconi = torch.cat([x_reconi[:,frameidx_t:,:,:,:], x_reconi[:,:frameidx_t,:,:,:]], 1)
			x_recon.append(x_reconi)

		x_recon = torch.stack(x_recon)

		self.vae.train()

		return x_recon, image, energy, coords_in, PE, KE, NSEP


	def image_to_seq(self, x_t, time=8):
		self.vae.eval()

		batchsize, _, channels, rows, columns = x_t.shape
		frameidx_t = torch.randint(0, x_t.shape[1]-1, (1,))
		image = x_t[:,frameidx_t,:,:,:]
		image = repeat(image, 'b 1 c w h -> b t c w h', t=time)
		if not self.config['model']['condnSonU'] and self.config['model']['dynamics'] in ['skewsymHamiltonian', 'Hamiltonian']:
			flag = True
			self.vae.lds.condnU = True
		else:
			flag = False

		rand_label = torch.linspace(0, self.config['actions']-1, self.config['actions'], dtype=int).to(x_t.device)
		u_t = torch.FloatTensor(len(rand_label), self.config['model']['u_dim']).to(x_t.device)
		u_t.zero_()
		u_t.scatter_(1, rand_label.unsqueeze(1), 1)
		x_recon = []
		for i, x_i in enumerate(image):
			x_i = repeat(x_i, 't c w h -> n t c w h', n=len(rand_label))	
			_, mu_z_i, _, q_t_i, mu_qt_i, _, _, mu_pt_i, _ = self.vae.encode(x_i, u_t)
			subspace_dim = q_t_i.shape[-1]
			if self.config['model']['dynamics'] == 'Fourier':
				s_t_gen_i, startidx = self.vae.dynamics(mu_qt_i, u_t.to(self.device[2]), time)
				s_t_gen_i = s_t_gen_i.to(self.device[0])
				subspace_dim = s_t_gen_i.shape[-1]
			elif self.config['model']['dynamics'] in ['RNN', 'Linear']:
				s_t_i = mu_qt_i
				s_t_gen_i = self.vae.lds(u_t.to(self.device[2]), s_t_i[:,frameidx_t,:,:].squeeze(1).to(self.device[2]), time, 0, test=True).to(self.device[0])
				#subspace_dim = mu_pt_i.shape[-1]
			else:
				s_t_i = torch.cat([mu_qt_i, mu_pt_i], 3)
				subspace_dim = mu_qt_i.shape[-1]
				#Concatenate Content and Motion Variables
				s_t_gen_i = self.vae.lds(u_t.to(self.device[2]), s_t_i[:,frameidx_t,:,:].squeeze(1).to(self.device[2]), time, frameidx_t).to(self.device[0])
			if self.config['model']['useZ']:
				mu_z_i = repeat(mu_z_i, 'b z -> b t z', t=time)
				z_q_gen_i = torch.cat([mu_z_i, rearrange(s_t_gen_i[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2)
			else:
				z_q_gen_i = rearrange(s_t_gen_i[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')

			x_reconi = self.vae.decode(z_q_gen_i)

			x_reconi = torch.cat([x_reconi[:,frameidx_t:,:,:,:], x_reconi[:,:frameidx_t,:,:,:]], 1)
			x_recon.append(x_reconi)
		x_recon = torch.stack(x_recon)
		if flag:
			self.vae.lds.condnU = False
			_, mu_z, _, q_t, mu_qt, _, _, mu_pt, _ = self.vae.encode(image, u_t)

			s_t = torch.cat([mu_qt, mu_pt], 3)
			s_t_gen = self.vae.lds(u_t.to(self.device[2]), s_t[:,frameidx_t,:,:].squeeze(1).to(self.device[2]), time, frameidx_t).to(self.device[0])

			if self.config['model']['useZ']:
				mu_z = repeat(mu_z, 'b z -> b t z', t=time)
				z_q_gen = torch.cat([mu_z, rearrange(s_t_gen[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2)
			else:
				z_q_gen = rearrange(s_t_gen[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')

			x_rec = self.vae.decode(z_q_gen)

			x_rec = torch.cat([x_rec[:,frameidx_t:,:,:,:], x_rec[:,:frameidx_t,:,:,:]], 1)
			x_recon = torch.cat([x_recon, x_rec.unsqueeze(1)],1)
		return x_recon, image


	def reconstruct_all_actions(self, x_t):
		self.vae.eval()

		rand_label = torch.linspace(0, self.config['actions']-1, self.config['actions'], dtype=int).to(x_t.device)
		u_t = torch.FloatTensor(len(rand_label), self.config['model']['u_dim']).to(x_t.device)
		u_t.zero_()
		u_t.scatter_(1, rand_label.unsqueeze(1), 1)
		if not self.config['model']['condnSonU'] and self.config['model']['dynamics'] in ['skewsymHamiltonian', 'Hamiltonian']:
			flag = True
			self.vae.lds.condnU = True
		else:
			flag = False

		x_recon, x_t_all = [], []
		timesteps = x_t.shape[1]
		for i, x_i in enumerate(x_t):
			x_i = repeat(x_i, 't c w h -> n t c w h', n=len(rand_label))

			_, mu_z_i, _, _, mu_qt_i, _, _, mu_pt_i, _ = self.vae.encode(x_i, u_t)
			subspace_dim = mu_qt_i.shape[-1]
			if self.config['model']['dynamics'] == 'Fourier':
				mu_st_i, frameidx_t = self.vae.dynamics(mu_qt_i, u_t.to(self.device[2]), timesteps)
				mu_st_i = mu_st_i.to(self.device[0])
				subspace_dim = mu_st_i.shape[-1]
				s_t_i_gen = mu_st_i
			elif self.config['model']['dynamics'] in ['RNN', 'Linear']:
				mu_st_i = mu_qt_i[:,0,:,:]
				s_t_i_gen, frameidx_t = self.vae.dynamics(mu_st_i, u_t, timesteps, test=True)
			else:
				mu_st_i = torch.cat([mu_qt_i, mu_pt_i], 3)
				s_t_i_gen, frameidx_t = self.vae.dynamics(mu_st_i, u_t, timesteps)

			#Concatenate Content and Motion Variables
			if self.config['model']['useZ']:
				mu_z_i = repeat(mu_z_i, 'b z -> b t z', t=timesteps)
				sz = torch.cat([mu_z_i, rearrange(s_t_i_gen[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2)
			else:
				sz = rearrange(s_t_gen_i[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')

			x_reconi = self.vae.decode(sz)
			
			x_reconi_idx = torch.cat([x_reconi[:,frameidx_t:,:,:,:], x_reconi[:,:frameidx_t,:,:,:]], 1)
			x_tidx = torch.cat([x_i[0,frameidx_t:,:,:,:], x_i[0,:frameidx_t,:,:,:]], 0).unsqueeze(0)
			x_recon.append(x_reconi_idx)
			x_t_all.append(x_tidx)

		x_recon = torch.stack(x_recon)
		x_t_all = torch.stack(x_t_all)
		if flag:
			self.vae.lds.condnU = False
			_, mu_z, _, _, mu_qt, _, _, mu_pt, _ = self.vae.encode(x_t, u_t)
			subspace_dim = mu_qt.shape[-1]
			s_t = torch.cat([mu_qt, mu_pt], 3)
			#Concatenate Content and Motion Variables
			s_t_gen = self.vae.lds(u_t.to(self.device[2]), s_t[:,frameidx_t,:,:].squeeze(1).to(self.device[2]), timesteps, frameidx_t).to(self.device[0])
			if self.config['model']['useZ']:
				mu_z = repeat(mu_z, 'b z -> b t z', t=timesteps)
				z_q_gen = torch.cat([mu_z, rearrange(s_t_gen[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2)
			else:
				z_q_gen = rearrange(s_t_gen[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')

			x_rec = self.vae.decode(z_q_gen)

			x_rec = torch.cat([x_rec[:,frameidx_t:,:,:,:], x_rec[:,:frameidx_t,:,:,:]], 1)

			x_recon = torch.cat([x_recon, x_rec.unsqueeze(1)],1)

		self.vae.train()
		return x_recon, x_t_all


	def sample_seq(self, u_t, mu_z=[], timesteps = 20):
		self.vae.eval()
		if len(mu_z) == 0:
			mu_z = torch.randn(len(u_t), self.config['model']['h_dim'])

		if self.config['model']['dynamics'] == 'Fourier':
			mu_st_pred = self.vae.dynamics(torch.randn(len(u_t), 1).to(u_t.device), u_t, timesteps)[0]
		else:
			mu_st_pred = self.vae.lds.sampling(u_t, timesteps, startidx=0)
		batchsize, timesteps, nm_subspaces, subspace_dim = mu_st_pred.shape
		if self.config['model']['useZ']:
			mu_z = repeat(mu_z, 'b z -> b t z', t=timesteps)
			z_s_t_gen = torch.cat([mu_z, rearrange(mu_st_pred[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2)
		else:
			z_s_t_gen = rearrange(mu_st_pred[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')

		x_recon = self.vae.decode(z_s_t_gen)
		return x_recon, z_q_t_gen


	def style_transfer(self, x_1, x_2, u_1, u_2):
		self.vae.eval()
		timesteps = x_1.shape[1]

		_, mu_z_1, _, _, mu_qt_1, _, _, mu_pt_1, _ = self.vae.encode(x_1, u_1)
		_, mu_z_2, _, _, mu_qt_2, _, _, mu_pt_2, _ = self.vae.encode(x_2, u_2)		


		if self.config['model']['useZ']:
			mu_z_1 = repeat(mu_z_1, 'b z -> b t z', t=timesteps)
			mu_z_2 = repeat(mu_z_2, 'b z -> b t z', t=timesteps)

			if self.config['model']['dynamics'] == 'Fourier':
				mu_qt_1 = self.vae.dynamics(mu_qt_1, u_1, timesteps)[0]
				mu_qt_2 = self.vae.dynamics(mu_qt_2, u_2, timesteps)[0]
			mu_qt_1 = rearrange(mu_qt_1, 'b t n v -> b t (n v)')
			mu_qt_2 = rearrange(mu_qt_2, 'b t n v -> b t (n v)')

			z_11 = torch.cat([mu_z_1, mu_qt_1], dim=2)
			z_12 = torch.cat([mu_z_1, mu_qt_2], dim=2)
			z_21 = torch.cat([mu_z_2, mu_qt_1], dim=2)
			z_22 = torch.cat([mu_z_2, mu_qt_2], dim=2)
		else:
			if self.config['model']['dynamics'] == 'Fourier':
				mu_qt_1 = self.vae.dynamics(mu_qt_1, u_1, timesteps)[0]
				mu_qt_2 = self.vae.dynamics(mu_qt_2, u_2, timesteps)[0]

			mu_qt_1 = rearrange(mu_qt_1, 'b t n v -> b t (n v)')
			mu_qt_2 = rearrange(mu_qt_2, 'b t n v -> b t (n v)')
			z_11, z_12, z_21, z_22 = mu_qt_1, mu_qt_2, mu_qt_1, mu_qt_2
			
		x_11 = self.vae.decode(z_11)
		x_22 = self.vae.decode(z_22)
		x_12 = self.vae.decode(z_12)
		x_21 = self.vae.decode(z_21)

		return x_11, x_22, x_12, x_21

	def motion_composition(self, x_1, x_2, u_1, u_2):
		self.vae.eval()
		timesteps = x_1.shape[1]

		_, mu_z_1, _, _, mu_qt_1, _, _, mu_pt_1, _ = self.vae.encode(x_1, u_1)
		_, mu_z_2, _, _, mu_qt_2, _, _, mu_pt_2, _ = self.vae.encode(x_2, u_2)		

		if self.config['model']['dynamics'] == 'Fourier':
			mu_qt_1 = self.vae.dynamics(mu_qt_1, u_1, timesteps)[0]
			mu_qt_2 = self.vae.dynamics(mu_qt_2, u_2, timesteps)[0]

		mv_1, mv_2, mv_12, mv_21 = mu_qt_1.clone(), mu_qt_2.clone(), mu_qt_1.clone(), mu_qt_2.clone()


		actions1, actions2 = u_1.argmax(1), u_2.argmax(1)
		batchsize, timesteps, nm_subspaces, subspace_dim = mv_1.shape
		if not self.config['model']['condnSonU']:
			mv_12 = 0.5*(mv_1 + mv_2)
			mv_21 = 0.5*(mv_1 + mv_2)
		else:
			mv_12[range(len(actions2)),:,actions2,:] = mv_2[range(len(actions2)),:,actions2,:]
			mv_21[range(len(actions1)),:,actions1,:] = mv_1[range(len(actions1)),:,actions1,:]


		mv_1 = rearrange(mv_1, 'b t n v -> b t (n v)')
		mv_2 = rearrange(mv_2, 'b t n v -> b t (n v)')
		mv_12 = rearrange(mv_12, 'b t n v -> b t (n v)')
		mv_21 = rearrange(mv_21, 'b t n v -> b t (n v)')

		if self.config['model']['useZ']:
			mu_z_1 = repeat(mu_z_1, 'b z -> b t z', t=timesteps)
			mu_z_2 = repeat(mu_z_2, 'b z -> b t z', t=timesteps)

			z_11 = torch.cat([mu_z_1, mv_1], dim=2)
			z_12 = torch.cat([mu_z_1, mv_12], dim=2)
			z_21 = torch.cat([mu_z_2, mv_1], dim=2)
			z_22 = torch.cat([mu_z_2, mv_21], dim=2)
	
		else:
			z_11, z_12, z_21, z_22 = mv_1, mv_12, mv_1, mv_21
		
		x_11 = self.vae.decode(z_11)
		x_22 = self.vae.decode(z_22)
		x_12 = self.vae.decode(z_12)
		x_21 = self.vae.decode(z_21)

		return x_11, x_22, x_12, x_21


	def random_invariant_sample(self, x_1, u_1):
		self.vae.eval()
		timesteps = x_1.shape[1]

		_, mu_z_1, _, _, mu_qt_1, _, _, mu_pt_1, _ = self.vae.encode(x_1, u_1)

		if self.config['model']['dynamics'] == 'Fourier':
			mu_qt_1 = self.vae.dynamics(mu_qt_1, u_1, timesteps)[0]
			
		ms_1 = rearrange(mu_qt_1, 'b t n v -> b t (n v)')
		if self.config['model']['useZ']:
			mu_z_1 = torch.randn(*mu_z_1.shape).to(mu_z_1.device)
			mu_z_1 = repeat(mu_z_1, 'b z -> b t z', t=timesteps)
			zq = torch.cat([mu_z_1, ms_1], dim=2)
		else:
			zq = ms_1
		x = self.vae.decode(zq)
		return x

	def random_variant_sample(self, x_1, u_1, timesteps=8):
		self.vae.eval()
		_, mu_z_1, _, _, mu_qt_1, _, _, mu_pt_1, _ = self.vae.encode(x_1, u_1)
		subspace_dim = mu_qt_1.shape[-1]
		if self.config['model']['dynamics'] == 'Fourier':
			mu_s_1 = self.vae.dynamics(torch.randn(*mu_qt_1.shape).to(mu_qt_1.device), u_1, timesteps)[0]
			subspace_dim = mu_s_1.shape[-1]
		else:
			mu_s_1 = self.vae.lds.sampling(u_1, timesteps, 0)
		if self.config['model']['useZ']:
			mu_z_1 = repeat(mu_z_1, 'b z -> b t z', t=timesteps)
			zq = torch.cat([mu_z_1, rearrange(mu_s_1[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')], dim=2)
		else:
			zq = rearrange(mu_s_1[:,:,:,:subspace_dim], 'b t n v -> b t (n v)')
		x = self.vae.decode(zq)
		return x


	def random_all_variant_sample(self, x_t, timesteps=8):
		self.vae.eval()
		
		rand_label = torch.linspace(0, self.config['actions']-1, self.config['actions'], dtype=int).to(x_t.device)
		u_t = torch.FloatTensor(len(rand_label), self.config['model']['u_dim']).to(x_t.device)
		u_t.zero_()
		u_t.scatter_(1, rand_label.unsqueeze(1), 1)
		startidx = 0
		if self.config['model']['dynamics'] == 'Fourier':
			mu_st = self.vae.dynamics(torch.randn(len(u_t),1), u_t, timesteps)[0]
			subspace_dim = mu_st.shape[-1]
		else:
			mu_st = self.vae.lds.sampling(u_t, timesteps, startidx)
			if self.config['model']['dynamics'] in ['Hamiltonian', 'Skewhamiltonian']:
				subspace_dim = int(mu_st.shape[-1]/2)
			else:
				subspace_dim = mu_st.shape[-1]

		mu_qt = mu_st[:,:,:,:self.vae.lds.subspace_dim]
		mu_qt = rearrange(mu_qt, 'b t n v -> b t (n v)')
		x_recon = []
		for i, x_i in enumerate(x_t):
			x_i = repeat(x_i, 't c w h -> n t c w h', n=len(rand_label))
			if self.config['model']['useZ']:
				_, mu_z_i, _, _, _, _, _, _, _ = self.vae.encode(x_i, u_t)
				mu_z_i = repeat(mu_z_i, 'b z -> b t z', t=timesteps)
				vz = torch.cat([mu_z_i, mu_qt], dim=2)
			else:
				vz = mu_qt
			x_reconi = self.vae.decode(vz)
			x_recon.append(x_reconi)
		return x_recon


class TrainerAE(nn.Module):
	def __init__(self, config, 
				data, 
				results_path,
				model_path,
				device):
		super(TrainerAE,self).__init__()
		self.config = config
		self.model_path = model_path
		self.results_path = results_path
		self.writer = SummaryWriter(f"{self.results_path}/summary")
		self.vae = VAE(self.config['model'], data, device_ids=device)
		self.optim = torch.optim.Adam(self.vae.parameters(), lr=self.config['optim']['lr'], 
								betas=(self.config['optim']['beta1'], 
								self.config['optim']['beta2']))


	def logPx(self, x, x_recon):
		_, time, ch, rows, cols = x.shape
		x = rearrange(x, 'b t c w h -> (b t) (c w h)')
		x_recon = rearrange(x_recon, 'b t c w h -> (b t) (c w h)')
		if self.config['reconloss'] == 'bce':
			return F.binary_cross_entropy(x_recon, x, reduction='sum')
		elif self.config['reconloss'] == 'l2':
			return F.mse_loss(x, x_recon, reduction='sum')
		elif self.config['reconloss'] == 'l1':
			return F.l1_loss(x, x_recon, reduction='sum')
		else:
			assert 0, f"Not impelemented {self.config['reconloss']}"

	def update(self, x_in):
		self.optim.zero_grad()
		batchsize, time, _, _, _ = x_in.shape
		x_recon, z, mu, logsigma = self.vae(x_in)
		_, _, z_dim = z.shape
		self.recon_loss = self.logPx(x_in, x_recon)
		mu = rearrange(mu, 'b t z -> (b t) z')
		logsigma = rearrange(logsigma, 'b t z -> (b t) z')
		self.kld = self.vae.vae_loss(mu, logsigma)
=		self.config['BetaVAE'] = 1.
		self.kld = self.kld/(batchsize*time)
		self.recon_loss = self.recon_loss/(batchsize*time)
		self.elbo_loss = (self.kld*self.config['BetaVAE'] + self.config['recon_term']*self.recon_loss)
		self.elbo_loss.backward()
		self.optim.step()

	def reconstruct(self, x_in):
		self.vae.eval()
		_, mu, logsigma = self.vae.encode(x_in)	
		x_recon = self.vae.decode(mu)
		return x_recon

	def save_model(self, epoch):
		gen = os.path.join(self.model_path, 'vae_{}.pt'.format(epoch+1))
		torch.save({'VAE':self.vae.state_dict(), 'optim':self.optim.state_dict()}, gen)

	def resume(self, checkpoint, device=torch.device('cpu')):
		state = torch.load(ckpt, map_location=device)
		self.vae.load_state_dict(state['VAE'])
		self.optim.load_state_dict(state['optim'])

	def summary(self, i):
		self.writer.add_scalar('Iter Loss/Reconstruction Loss', self.recon_loss, i)
		self.writer.add_scalar('Iter Loss/Variational Loss', self.kld, i)
		self.writer.add_scalar('Iter Loss/Total Loss', self.elbo_loss, i)

	def update_lr(self):
		self.scheduler.step()
