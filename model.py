
import torch.nn.functional as F
import torch
import torch.nn as nn
from base import Reshape, conv_shape
import torch.autograd as autograd
import pdb
import numpy as np
import torch.nn.init as init
from einops import rearrange, repeat
from base import EncoderConv, DecoderConv, \
				EncoderConv1, DecoderConv1, \
				EncoderConv2, DecoderConv2, \
				EncoderResNet, DecoderResNet, \
				ContentMotion, DynamicalModel, \
				Discriminator_V, TemporalBlock, \
				EncoderMNIST, DecoderMNIST

class SubspaceVAE(nn.Module):
	def __init__(self, config, data, device_ids=[0,1,2]):
		super(SubspaceVAE, self).__init__()
		self.config = config
		self.device_ids = device_ids
		if self.config['dynamics'] == 'symplecticform':
			decoder_dim_V = int(self.config['v_dim']/2)
		else:
			decoder_dim_V = self.config['v_dim']

		if not self.config['useZ']:
			self.config['z_dim'] = 0

		if data == 'rmnist':
			self.encoder = EncoderMNIST(self.config['channels'],
							self.config['h_dim']).to(self.device_ids[0])
			self.decoder = DecoderMNIST(self.config['h_dim'],
							decoder_dim_V,
							self.config['z_dim'],
							self.config['channels']).to(self.device_ids[1])

		else:
			if self.config['network'] == 'conv':
				self.encoder = EncoderConv(self.config['channels'], 
								self.config['h_dim']).to(self.device_ids[0])

				self.decoder = DecoderConv(self.config['h_dim'], 
								decoder_dim_V, 
								self.config['z_dim'],
								self.config['channels']).to(self.device_ids[1])

			elif self.config['network'] == 'conv1':
				self.encoder = EncoderConv1(self.config['channels'], 
								self.config['h_dim']).to(self.device_ids[0])

				self.decoder = DecoderConv1(self.config['h_dim'], 
								decoder_dim_V, 
								self.config['z_dim'],
								self.config['channels']).to(self.device_ids[1])

			elif self.config['network'] == 'conv2':
				self.encoder = EncoderConv2(self.config['channels'], 
								self.config['h_dim']).to(self.device_ids[0])

				self.decoder = DecoderConv2(self.config['h_dim'], 
								decoder_dim_V, 
								self.config['z_dim'],
								self.config['channels']).to(self.device_ids[1])

			elif self.config['network'] == 'resnetladder':
				self.encoder = EncoderResNet(self.config['width'],
								self.config['height'],
								self.config['channels'],
								self.config['h_dim']).to(self.device_ids[0])

				self.decoder = DecoderResNet(self.config['width'],
								self.config['height'],
								self.config['channels'],
								self.config['h_dim'],
								decoder_dim_V,
								self.config['z_dim']).to(self.device_ids[1])

			else:
				assert 0, f"Network Not implemented {self.config['network']}"

		if self.config['sequential']:
			self.contentMotion = ContentMotion(self.config['h_dim'], 
							self.config['z_dim'],
							self.config['v_dim'],
							self.config['u_dim'],
							self.config['uselstmZ'],
							self.config['condnSonU'],
							self.config['useZ'],
							self.config['dynamics']).to(self.device_ids[0])
			if self.config['dynamics'] in ['Hamiltonian', 'SkewHamiltonian']:
				self.W_p = TemporalBlock(self.config['v_dim'], self.config['v_dim'], 
							kernel_size=(4,), dropout=0, padding=3, dilation=1, 
							stride=1).to(self.device_ids[0])
		
				self.mu_p = nn.Linear(self.config['v_dim'], self.config['v_dim']).to(self.device_ids[0])
				self.sigma_p = nn.Linear(self.config['v_dim'], self.config['v_dim']).to(self.device_ids[0])
				self.mu_p.apply(self.weights_init(init_type='orthogonal')) 
				self.sigma_p.apply(self.weights_init(init_type='orthogonal'))
			#if self.config['useZ']:
			#	self.mu_z = nn.Linear(self.config['z_dim'], self.config['z_dim']).to(self.device_ids[0])
			#	self.sigma_z = nn.Linear(self.config['z_dim'], self.config['z_dim']).to(self.device_ids[0])
			
			if self.config['dynamics'] == 'Fourier':
				in_dim, out_dim = 1, 1
			else:
				in_dim, out_dim = self.config['v_dim'], self.config['v_dim']

			self.mu_q = nn.Linear(in_dim, out_dim).to(self.device_ids[0])
			self.sigma_q = nn.Linear(in_dim, out_dim).to(self.device_ids[0])
			self.mu_q.apply(self.weights_init(init_type='orthogonal'))
			self.sigma_q.apply(self.weights_init(init_type='orthogonal'))


			if not self.config['condnSonU']:
				self.config['u_dim'] = self.config['nm_operators']

			if self.config['dynamics'] in ['Hamiltonian', 'SkewHamiltonian', 'RNN', 'Linear']:
				self.lds = DynamicalModel(self.config['v_dim'], 
								self.config['u_dim'], 
								self.config['dynamics'],
								self.config['condnSonU'],
								self.config['projection']).to(self.device_ids[2])

		else:
			self.latent_layer = nn.Sequential(
					nn.Linear(self.config['h_dim'], self.config['z_dim']),
					nn.BatchNorm1d(self.config['z_dim']),
					nn.LeakyReLU(0.2),
				).to(self.device_ids[0])

		if self.config['useZ']:
			self.mu_z = nn.Linear(self.config['z_dim'], self.config['z_dim']).to(self.device_ids[0])
			self.sigma_z = nn.Linear(self.config['z_dim'], self.config['z_dim']).to(self.device_ids[0])
			self.mu_z.apply(self.weights_init(init_type='orthogonal'))
			self.sigma_z.apply(self.weights_init(init_type='orthogonal'))
		self.encoder.apply(self.weights_init(init_type='orthogonal'))
		self.decoder.apply(self.weights_init(init_type='orthogonal'))


	def weights_init(self, init_type='kaiming'):
		def init_fun(m):
			classname = m.__class__.__name__
			if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
				if init_type == 'gaussian':
					init.normal_(m.weight.data, 0.0, 0.02)
				elif init_type == 'xavier':
					init.xavier_normal_(m.weight.data, gain=np.sqrt(2))
				elif init_type == 'kaiming':
					init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
				elif init_type == 'orthogonal':
					init.orthogonal_(m.weight.data, gain=np.sqrt(2))
				elif init_type == 'default':
					pass
				else:
					assert 0, f"Initialization not implemented {init_type}"
				if hasattr(m, 'bias') and m.bias is not None:
					init.constant_(m.bias.data, 0.0)
		return init_fun


	def reparameterisation(self, x, space='v'):
		if space == 'q':
			mu, logsig = self.mu_q(x), self.sigma_q(x)
		elif space == 'z':
			mu, logsig = self.mu_z(x), self.sigma_z(x)
		elif space == 'p':
			mu, logsig = self.mu_p(x), self.sigma_p(x)
		else:
			assert 0, f"Not implemented {space}"

		noise = torch.randn(*mu.shape).to(mu.device)
		sig = torch.exp(0.5*logsig)
		z = mu + noise * sig
		return z, mu, logsig

	def pretrainEncode(self, x):
		h_t = self.encoder(x)
		batch, time, z_dim = h_t.shape
		h_t = rearrange(h_t, 'b t d -> (b t) d')
		h_t = self.latent_layer(h_t)
		z, mu_t, logsigma_t = self.reparameterisation(h_t, space='z')
		z = rearrange(z, '(b t) d -> b t d', b=batch, t=time)
		mu_t = rearrange(mu_t, '(b t) d -> b t d', b=batch, t=time)
		logsigma_t = rearrange(logsigma_t, '(b t) d -> b t d', b=batch, t=time)
		return z, mu_t, logsigma_t

	def splitSubspace(self, v, mu_v, logsigma_v, batch, time):
		v = rearrange(v, '(b t) v -> b t v', b=batch, t=time)
		mu_v = rearrange(mu_v, '(b t) v -> b t v', b=batch, t=time)
		logsigma_v = rearrange(logsigma_v, '(b t) v -> b t v', b=batch, t=time)
		v = torch.stack(v.chunk(self.config['u_dim'], dim=2)).transpose(1,0).transpose(2,1)
		mu_v = torch.stack(mu_v.chunk(self.config['u_dim'], dim=2)).transpose(1,0).transpose(2,1)
		logsigma_v = torch.stack(logsigma_v.chunk(self.config['u_dim'], dim=2)).transpose(1,0).transpose(2,1)
	
		v = v/(v.norm(dim=3).unsqueeze(3) + 1e-8)
		return v, mu_v, logsigma_v

	def encode(self, x, u_t=None):
		h_t = self.encoder(x)
		batch, time, h_dim = h_t.shape
		if self.config['useZ']:
			z, v_t = self.contentMotion(h_t, u_t)
			z, mu_z, logsigma_z = self.reparameterisation(z, space='z')
		else:
			_, v_t = self.contentMotion(h_t, u_t)
			z, mu_z, logsigma_z = None, None, None
		
		if self.config['dynamics'] == 'Fourier':
			q_t = rearrange(v_t, 'b t d -> (b t) d') 
			q_t, mu_qt, logsigma_qt = self.reparameterisation(q_t, space='q')
			q_t = rearrange(q_t, '(b t) d -> b t d', b=batch, t=time)

			mu_qt = rearrange(mu_qt, '(b t) d -> b t d', b=batch, t=time)
			logsigma_qt = rearrange(logsigma_qt, '(b t) d -> b t d', b=batch, t=time)
			p_t, mu_pt, logsigma_pt = None, None, None

		elif self.config['dynamics'] in ['Linear', 'RNN']:
			q_t = rearrange(v_t, 'b t d -> (b t) d')
			q_t, mu_qt, logsigma_qt = self.reparameterisation(q_t, space='q')
			if self.config['condnSonU']:
				q_t, mu_qt, logsigma_qt = self.splitSubspace(q_t, mu_qt, logsigma_qt, batch, time)
			else:
				q_t = rearrange(q_t, '(b t) d -> b t 1 d', b=batch, t=time)
				mu_qt = rearrange(mu_qt, '(b t) d -> b t 1 d', b=batch, t=time)
				logsigma_qt = rearrange(logsigma_qt, '(b t) d -> b t 1 d', b=batch, t=time)

			p_t, mu_pt, logsigma_pt = None, None, None
		#	p_t = rearrange(self.W_p(rearrange(v_t, 'b t d -> b d t')), 'b d t -> b t d')
		#	p_t = rearrange(p_t, 'b t d -> (b t) d')		
		#	p_t, mu_pt, logsigma_pt = self.reparameterisation(p_t, space='p')
		#	p_t, mu_pt, logsigma_pt = self.splitSubspace(p_t, mu_pt, logsigma_pt, batch, time)
		#	q_t, mu_qt, logsigma_qt = None, None, None
		else:
			p_t = rearrange(self.W_p(rearrange(v_t, 'b t d -> b d t')), 'b d t -> b t d')
			p_t = rearrange(p_t, 'b t d -> (b t) d')
			p_t, mu_pt, logsigma_pt = self.reparameterisation(p_t, space='p')
			
			q_t = rearrange(v_t, 'b t d -> (b t) d')
			q_t, mu_qt, logsigma_qt = self.reparameterisation(q_t, space='q')
		

			q_t, mu_qt, logsigma_qt = self.splitSubspace(q_t, mu_qt, logsigma_qt, batch, time)
			p_t, mu_pt, logsigma_pt = self.splitSubspace(p_t, mu_pt, logsigma_pt, batch, time)

		if self.config['condnSonU'] and self.config['projection']:
			projection = torch.zeros(*q_t.shape).to(x.device)
			actions = u_t.argmax(axis=1)
			projection[range(len(actions)),:,actions,:] = 1.

			if self.config['dynamics'] in ['RNN', 'Linear']:
				q_t = projection * q_t
				mu_qt = projection * mu_qt
				logsigma_qt = projection * logsigma_qt
			elif self.config['dynamics'] in ['Hamiltonian', 'SkewHamiltonian']:
				q_t = projection * q_t
				mu_qt = projection * mu_qt
				logsigma_qt = projection * logsigma_qt

				p_t = projection * p_t
				mu_pt = projection * mu_pt
				logsigma_pt = projection * logsigma_pt
			else:
				pass
		return z, mu_z, logsigma_z, q_t, mu_qt, logsigma_qt, p_t, mu_pt, logsigma_pt


	def dynamics(self, v_t, u_t, timesteps, test=False):
		if self.config['dynamics'] == 'Fourier':
			startidx = 2
			T_i =  torch.linspace(0, 1, timesteps-startidx).to(v_t.device)
			T_i = v_t[:,startidx,0].reshape(-1,1) + T_i
			T_i = torch.cat([v_t[:,:startidx,0], T_i], 1)
			c_d = 2**(torch.linspace(0, int(self.config['v_dim']/2)-1, int(self.config['v_dim']/2)))
			coords = torch.einsum('b t, d -> b t d', T_i, c_d.to(v_t.device))
			v_t_seq = torch.cat([torch.sin(coords), torch.cos(coords)], 2)
			v_t_seq = repeat(v_t_seq, 'b t v -> b t n v', n=1)
		# Generate trajectory using Hamiltonian Operators
		elif self.config['dynamics'] in ['RNN', 'Linear']:
			startidx = 0
			v_t_seq = self.lds(u_t.to(self.device_ids[2]), v_t.to(self.device_ids[2]), timesteps, startidx, test)
		elif self.config['qv_x'] == 'qv_f_x1toT':
			startidx = torch.randint(0, v_t.shape[1]-1, (1,))
			v_f = v_t[:,startidx,:,:].squeeze(1)
			v_t_seq = self.lds(u_t.to(self.device_ids[2]), v_f.to(self.device_ids[2]), timesteps, startidx)
		elif self.config['qv_x'] == 'qv_x0':
			startidx = 0
			v_f = v_t[:,0,:,:]
			v_t_seq = self.lds(u_t.to(self.device_ids[2]), v_f.to(self.device_ids[2]), timesteps, startidx)
		else:
			assert 0,f"Not impelemented {self.config['qv_x']}"

		v_t_seq = v_t_seq.to(self.device_ids[0])
		return v_t_seq, startidx

	def decode(self, z):
		return self.decoder(z.to(self.device_ids[1])).to(self.device_ids[0])

	def vae_loss(self, mu, logsigma):
		kld = -0.5 * torch.sum(1 +logsigma - mu.pow(2) - logsigma.exp())
		return kld

	def recon_loss(self, x, x_recon):
		if self.config['losstype'] == 'mse':
			recon_loss = F.mse_loss(x, x_recon, reduction='sum')
		elif self.config['losstype'] == 'bce':
			recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
		else:
			assert 0, f"{self.config['losstype']} not implemented"
		return recon_loss



class VAE(nn.Module):
	def __init__(self, config, data, device_ids=[2,3]):
		super(VAE, self).__init__()
		self.config = config
		self.device_ids = device_ids	
		if data in ['kth','sprites', 'MUG']:
			if self.config['network'] == 'conv':
				self.encoder = EncoderConv(self.config['channels'], 
								self.config['h_dim'],  
								self.config['ht_dim'],
								False,
								sequential=False).to(self.device_ids[0])

				self.decoder = DecoderConv(self.config['ht_dim'], 
								self.config['z_dim'], 
								self.config['channels'], 
								self.config['h_dim']).to(self.device_ids[1])
			elif self.config['network'] == 'conv1':
				self.encoder = EncoderConv1(self.config['channels'], 
								self.config['h_dim'],  
								self.config['ht_dim'],
								False,
								sequential=False).to(self.device_ids[0])

				self.decoder = DecoderConv1(self.config['ht_dim'], 
								self.config['z_dim'], 
								self.config['channels'], 
								self.config['h_dim']).to(self.device_ids[1])
			elif self.config['network'] == 'conv2':
				self.encoder = EncoderConv2(self.config['channels'], 
								self.config['h_dim'],  
								self.config['ht_dim'],
								False,
								sequential=False).to(self.device_ids[0])

				self.decoder = DecoderConv2(self.config['ht_dim'], 
								self.config['z_dim'], 
								self.config['channels'], 
								self.config['h_dim']).to(self.device_ids[1])

			elif self.config['network'] == 'resnetladder':
				self.encoder = EncoderResNet(self.config['width'],
								self.config['height'],
								self.config['channels'],
								self.config['h_dim'],
								self.config['ht_dim'],
								False,
								sequential=False).to(self.device_ids[0])
				self.decoder = DecoderResNet(self.config['width'],
								self.config['height'],
								self.config['channels'],
								self.config['ht_dim'],
								self.config['z_dim'],
								self.config['h_dim']).to(self.device_ids[1])

			else:
				assert 0, f"Network Not implemented {self.config['network']}"
		else:
			assert 0, f"Not implemented {self.config['data']}"

		self.mu_t = nn.Linear(self.config['h_dim'], self.config['h_dim'] + self.config['z_dim']).to(self.device_ids[0])
		self.mu_t.apply(self.weights_init(init_type='kaiming'))

		self.sigma_t = nn.Linear(self.config['h_dim'], self.config['h_dim'] + self.config['z_dim']).to(self.device_ids[0])
		self.sigma_t.apply(self.weights_init(init_type='kaiming'))

		self.encoder.apply(self.weights_init(init_type='kaiming'))
		self.decoder.apply(self.weights_init(init_type='kaiming'))

	def weights_init(self, init_type='kaiming'):
		def init_fun(m):
			classname = m.__class__.__name__
			if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
				if init_type == 'gaussian':
					init.normal_(m.weight.data, 0.0, 0.02)
				elif init_type == 'xavier':
					init.xavier_normal_(m.weight.data, gain=np.sqrt(2))
				elif init_type == 'kaiming':
					init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
				elif init_type == 'orthogonal':
					init.orthogonal_(m.weight.data, gain=np.sqrt(2))
				elif init_type == 'default':
					pass
				else:
					assert 0, f"Initialization not implemented {init_type}"
				if hasattr(m, 'bias') and m.bias is not None:
					init.constant_(m.bias.data, 0.0)
		return init_fun


	def forward(self, x):
		z_t, mu_t, logsigma_t = self.encode(x)
		x_recon = self.decode(z_t)
		return x_recon, z_t, mu_t, logsigma_t

	def reparameterization(self, x):
		mu = self.mu_t(x)
		logsig = self.sigma_t(x)
		noise = torch.randn(*mu.shape).to(mu.device)
		sig = torch.exp(0.5*logsig)
		z = mu + noise * sig
		return z, mu, logsig

	def encode(self, x):
		h_t = self.encoder(x)
		batch, time, hdim = h_t.shape
		h_t = h_t.view(-1, hdim)
		z_t, mu_t, logsigma_t = self.reparameterization(h_t)
		zdim = z_t.shape[1]
		z_t = z_t.view(batch, time, zdim)
		mu_t = mu_t.view(batch, time, zdim)
		logsigma_t = logsigma_t.view(batch, time, zdim)
		return z_t, mu_t, logsigma_t

	def decode(self, z):
		return self.decoder(z.to(self.device_ids[1])).to(self.device_ids[0])


	def vae_loss(self, mu, logsigma):
		#kld = 0.0
		kld = torch.mean(-0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=1), dim=0)
		return kld

	def vae_loss_T(self, mu, logsigma):
		kld = 0.0
		for i in range(mu.shape[1]):
			kld += -0.5 * torch.sum(1 + logsigma[:,i,:] - mu[:,i,:].pow(2) - logsigma[:,i,:].exp())
		return kld

	def recon_loss(self, x, x_recon):
		if self.config['losstype'] == 'mse':
			recon_loss = F.mse_loss(x, x_recon, reduction='sum')
		elif self.config['losstype'] == 'bce':
			recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
		else:
			assert 0, f"{self.config['losstype']} not implemented"
		return recon_loss
