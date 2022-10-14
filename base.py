import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init
import pdb

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

def conv_shape(in_width, in_height, kernel, stride, padding):
	out_height = int((in_height + 2*padding - kernel[0])/stride + 1)
	out_width = int((in_width + 2*padding - kernel[1])/stride + 1)
	return out_height, out_width

class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()
		
class Reshape(nn.Module):
	def __init__(self, shape):
		super().__init__()
		self.shape = shape

	def forward(self, x):
		return x.reshape(*self.shape)

class Permute(nn.Module):
	def __init__(self, axis):
		super().__init__()
		self.axis = axis

	def forward(self, x):
		return x.permute(*self.axis)


class LinearLayer(nn.Module):
	def __init__(self, in_dim, 
						out_dim, 
						norm=False, 
						activation='lrelu'):
		super(LinearLayer, self).__init__()
	
		self.affine = nn.Linear(in_dim, out_dim)
		if norm:
			self.norm = nn.BatchNorm1d(out_dim)
		else:
			self.norm = None

		if activation=='relu':
			self.activation = nn.ReLU(inplace=True)
		elif activation=='lrelu':
			self.activation = nn.LeakyReLU(0.2, inplace=True)
		elif activation=='tanh':
			self.activation = nn.Tanh()
		elif activation=='sigmoid':
			self.activation = nn.Sigmoid()
		elif activation=='selu':
			self.activation = nn.SELU(inplace=True)
		elif activation=='softmax':
			self.activation = nn.Softmax(dim=1)
		elif activation == 'none':
			self.activation = None
		else:
			assert 0,'Unsupported activation {}'.format(activation)

	def forward(self, x):
		out = self.affine(x)
		if self.norm:
			out = self.norm(out)
		if self.activation:
			out = self.activation(out)
		return out


class ConvLayer(nn.Module):
	def __init__(self, in_channels, 
						out_channels, 
						kernel, 
						stride, 
						pad=0,
						norm=False, 
						activation='lrelu', 
						pad_type='zero'):
		super(ConvLayer, self).__init__()
		if pad_type == 'reflect':
			self.pad = nn.ReflectionPad2d(pad)
		elif pad_type == 'replicate':
			self.pad = nn.ReplicationPad2d(pad)
		elif pad_type == 'zero':
			self.pad = nn.ZeroPad2d(pad)
		else:
			assert 0, 'Unsupported padding type: {}'.format(pad_type)
	
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride)
		if norm:
			self.norm = nn.BatchNorm2d(out_channels)
		else:
			self.norm = None

		if activation=='relu':
			self.activation = nn.ReLU(inplace=True)
		elif activation=='lrelu':
			self.activation = nn.LeakyReLU(0.2, inplace=True)
		elif activation=='tanh':
			self.activation = nn.Tanh()
		elif activation=='sigmoid':
			self.activation = nn.Sigmoid()
		elif activation=='selu':
			self.activation = nn.SELU(inplace=True)
		elif activation=='softmax':
			self.activation = nn.Softmax(dim=1)
		elif activation == 'none':
			self.activation = None
		else:
			assert 0,'Unsupported activation {}'.format(activation)

	def forward(self, x):
		out = self.conv(self.pad(x))
		if self.norm:
			out = self.norm(out)
		if self.activation:
			out = self.activation(out)
		return out


class ConvTransposeLayer(nn.Module):
	def __init__(self, in_channels, 
				out_channels, 
				kernel, 
				stride, 
				norm=False, 
				activation='lrelu', 
				pad_type='zero', 
				pad=0):
		super(ConvTransposeLayer, self).__init__()

		if pad_type == 'reflect':
			self.pad = nn.ReflectionPad2d(pad)
		elif pad_type == 'replicate':
			self.pad = nn.ReplicationPad2d(pad)
		elif pad_type == 'zero':
			self.pad = nn.ZeroPad2d(pad)
		else:
			assert 0, 'Unsupported padding type: {}'.format(pad_type)

		self.convt = nn.ConvTranspose2d(in_channels, out_channels, 
							kernel_size=kernel, stride=stride)
		if norm:
			self.norm = nn.BatchNorm2d(out_channels)
		else:
			self.norm = None

		if activation=='relu':
			self.activation = nn.ReLU(inplace=True)
		elif activation=='lrelu':
			self.activation = nn.LeakyReLU(0.2, inplace=True)
		elif activation=='tanh':
			self.activation = nn.Tanh()
		elif activation=='sigmoid':
			self.activation = nn.Sigmoid()
		elif activation=='selu':
			self.activation = nn.SELU(inplace=True)
		elif activation=='softmax':
			self.activation = nn.Softmax(dim=1)
		elif activation == 'none':
			self.activation = None
		else:
			assert 0,'Unsupported activation {}'.format(activation)

	def forward(self, x):
		out = self.convt(self.pad(x))
		if self.norm:
			out = self.norm(out)
		if self.activation:
			out = self.activation(out)
		return out

class ResLayer2D(nn.Module):
	def __init__(self, in_channels, 
				out_channels,
				kernel, 
				norm=False,
				sample='up',
				activation='relu', 
				channeldrop=False,
				pad_type='zero'):
		super(ResLayer2D, self).__init__()
		self.out_channels = out_channels
		res = []
		res.append(ConvLayer(out_channels, out_channels, kernel, 1, 1, norm=norm, activation=activation, pad_type=pad_type))
		res.append(ConvLayer(out_channels , out_channels, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type))
		self.res = nn.Sequential(*res)
		self.channeldrop = channeldrop
		self.channel_diff = in_channels - out_channels
		self.sample = sample
		if self.sample == 'up':
			self.samplelayer = nn.Upsample(scale_factor=2)
		elif self.sample == 'down':
			self.samplelayer = nn.AvgPool2d(2)
		else:
			assert 0, "Sampling {sample} not identified. Choose one of the following: up or down"

		self.project_x = nn.Sequential(
						nn.BatchNorm2d(in_channels),
						nn.ReLU(inplace=True),
						nn.Conv2d(in_channels, out_channels, 1, 1, 0),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(inplace=True),
					)
		if not self.channeldrop:
			if self.channel_diff<0:
				self.channel_layer = nn.Conv2d(in_channels, abs(self.channel_diff), 1, 1, 0)			
			else:
				self.channel_layer = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
		
							
	def forward(self, x):
		h = self.project_x(x)
		if self.channeldrop and self.channel_diff>0:
			x = x[:, :self.out_channels]
			x = self.samplelayer(x)
		else:
			x = self.samplelayer(x)
			if self.channel_diff<0:
				x = torch.cat([x, self.channel_layer(x)], 1)
			else:
				x = self.channel_layer(x)	
		h = self.samplelayer(h)
		out = self.res(h)
		out += x
		return out


class SelfAttention(nn.Module):
	def __init__(self, in_dim):
		super(SelfAttention,self).__init__()
		self.chanel_in = in_dim
		self.query = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size= 1, bias=False)
		self.key = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1, bias=False)
		self.value = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//2 , kernel_size= 1, bias=False)
		self.out = nn.Conv2d(in_channels = in_dim//2 , out_channels = in_dim , kernel_size= 1, bias=False)
		self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
		
		self.query.apply(self.weights_init(init_type='xavier_uf'))
		self.key.apply(self.weights_init(init_type='xavier_uf')) 
		self.value.apply(self.weights_init(init_type='xavier_uf'))

	def weights_init(self, init_type='kaiming'):
		def init_fun(m):
			classname = m.__class__.__name__
			if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
				if init_type == 'gaussian':
					init.normal_(m.weight.data, 0.0, 0.02)
				elif init_type == 'xavier':
					init.xavier_normal_(m.weight.data, gain=np.sqrt(2))
				elif init_type == 'xavier_uf':
					init.xavier_uniform_(m.weight.data)
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
		batchsize, channels, width, height = x.size()
		proj_query  = self.query(x)
		proj_key =  F.max_pool2d(self.key(x), [2,2])
		proj_value = F.max_pool2d(self.value(x), [2,2])

		proj_query = rearrange(proj_query, 'b c w h -> b c (w h)')
		proj_key = rearrange(proj_key, 'b c w h -> b c (w h)')
		proj_value = rearrange(proj_value, 'b c w h -> b c (w h)')

		weights = F.softmax(torch.einsum('b c f, b c k -> b f k', proj_key, proj_query), -1)

		out = torch.einsum('b c f, b f k -> b c k', proj_vale, attention)
		out = rearrange(out, 'b c (w h)-> b c w h', w=width, h=height)

		out = self.gamma*self.out(out) + x
		return out


class ContentMotion(nn.Module):
	def __init__(self, in_dim, 
				z_dim,
				v_dim,
				u_dim,
				uselstmZ=False,
				condnU=True,
				useZ=True,
				dynamics='Hamiltonian'):
		super(ContentMotion, self).__init__()
		self.in_dim = in_dim
		self.z_dim = z_dim
		self.v_dim = v_dim
		self.u_dim = u_dim

		self.condnU = condnU
		self.useZ = useZ
		self.uselstmZ = uselstmZ
		self.dynamics = dynamics

		if self.useZ:
			if self.uselstmZ:
				self.lstm = nn.LSTM(self.in_dim, int(self.z_dim/2), 2, bidirectional=True, batch_first=True)
			else:
				self.fc_layer = nn.Sequential(
						nn.Linear(self.in_dim, self.z_dim),
						nn.BatchNorm1d(self.z_dim),
						nn.LeakyReLU(0.2),
					)

		if self.condnU:
			self.subspaceV = nn.Sequential(
							nn.Linear(self.in_dim + self.u_dim, 2*self.v_dim),
							nn.BatchNorm1d(2*self.v_dim),
							nn.LeakyReLU(0.2),
							nn.Linear(2*self.v_dim, self.v_dim),
							nn.BatchNorm1d(self.v_dim),
							nn.LeakyReLU(0.2),
						)
		else:
			if self.dynamics == 'Fourier':
				self.spaceV = nn.Sequential(
								nn.Linear(self.in_dim, 2*self.v_dim),
								nn.BatchNorm1d(2*self.v_dim),
								nn.LeakyReLU(0.2),
								nn.Linear(2*self.v_dim, 1),
								nn.Tanh(),
						)
				self.temp = TemporalBlock(1, 1,	kernel_size=(3,), dropout=0, 
								padding=2, dilation=1, 
								stride=1)

			else:
				self.spaceV = nn.Sequential(
								nn.Linear(self.in_dim, 2*self.v_dim),
								nn.BatchNorm1d(2*self.v_dim),
								nn.LeakyReLU(0.2),
								nn.Linear(2*self.v_dim, self.v_dim),
								nn.BatchNorm1d(self.v_dim),
								nn.LeakyReLU(0.2),
							)

	def forward(self, h_t, u_t=None):
		batch, time, _ = h_t.shape
		if self.condnU:
			u_t = repeat(u_t, 'b u -> b t u', t=time)
			hv_t = torch.cat([h_t, u_t], dim=2)
			hv_t = rearrange(hv_t, 'b t hv -> (b t) hv')
			v_t = self.subspaceV(hv_t)
			v_t = rearrange(v_t, '(b t) v -> b t v', b=batch, t=time)
		else:
			if self.dynamics=='Fourier':
				#T_i = torch.linspace(0, 1, time).to(h_t.device)
				#c_d = 2**(torch.linspace(0, int(self.v_dim/2)-1, int(self.v_dim/2)))*np.pi
				#c_d = c_d.to(h_t.device)
				#coords = torch.einsum('t, d -> t d', T_i, c_d)
				#v_t = torch.cat([torch.sin(coords), torch.cos(coords)], 1)
				#v_t = repeat(v_t, 't d -> b t d', b=batch)

				hv_t = rearrange(h_t, 'b t h -> (b t) h')
				v_t = self.spaceV(hv_t)
				v_t = rearrange(v_t, '(b t) v -> b t v', b=batch, t=time)
				v_t = rearrange(self.temp(rearrange(v_t, 'b t d -> b d t')), 'b d t -> b t d')
			else:
				hv_t = rearrange(h_t, 'b t h -> (b t) h')
				v_t = self.spaceV(hv_t)
				v_t = rearrange(v_t, '(b t) v -> b t v', b=batch, t=time)
		if self.useZ:
			if self.uselstmZ:
				s_t, _ = self.lstm(h_t)
				forward = s_t[:, -1, 0:int(self.z_dim/2)]
				backward = s_t[:, 0, int(self.z_dim/2):self.z_dim]
				z = torch.cat([forward, backward], dim=1)
			else:
				s_t = h_t.mean(dim=1)
				z = self.fc_layer(s_t)
			return z, v_t
		else:
			return None, v_t


class EncoderMNIST(nn.Module):
	def __init__(self, channels, 
				h_dim):
		super(EncoderMNIST, self).__init__()
		self.h_dim = h_dim
		self.enc = nn.Sequential(
				nn.Conv2d(channels, 128, 5, 2, 2),  
				nn.BatchNorm2d(128),
				nn.ReLU(),
				nn.Conv2d(128, 128, 5, 2, 2),         
				nn.BatchNorm2d(128),
				nn.ReLU(),
				nn.Conv2d(128, 128, 5, 2, 2),          
				nn.BatchNorm2d(128),
				nn.ReLU(),
				Rearrange('b c w h -> b (c w h)'),
				nn.Linear(128*4*4, 4096),
				nn.BatchNorm1d(4096),
				nn.ReLU(),
				nn.Linear(4096, self.h_dim),
				nn.BatchNorm1d(self.h_dim),
				nn.LeakyReLU(),
			)

	def forward(self, x):
		batch, time, channels, rows, width = x.shape
		x = rearrange(x, 'b t c r w -> (b t) c r w')
		x_t = self.enc(x)
		x_t = rearrange(x_t, '(b t) z -> b t z', b=batch, t=time)
		return x_t


class DecoderMNIST(nn.Module):
	def __init__(self, h_dim,
			v_dim,
			z_dim,
			out_channels):
		super(DecoderMNIST, self).__init__()
		self.out_channels = out_channels
		in_dim = z_dim + v_dim
		self.dec = nn.Sequential(
		nn.Linear(in_dim, h_dim),
		nn.BatchNorm1d(h_dim),
		nn.ReLU(),
		nn.Linear(h_dim, 128*4*4),
		nn.BatchNorm1d(128*4*4),
		nn.ReLU(),
		Rearrange('b (c w h) -> b c w h', c=128, w=4, h=4),
		nn.ConvTranspose2d(128, 128, 3, 1, 0),
		nn.BatchNorm2d(128),
		nn.ReLU(),
		nn.ConvTranspose2d(128, 128, 5, 2, 1),
		nn.BatchNorm2d(128),
		nn.ReLU(),
		nn.ConvTranspose2d(128, 128, 5, 2, 1, 1),
		nn.BatchNorm2d(128),
		nn.ReLU(),
		nn.ConvTranspose2d(128, self.out_channels, 5, 1, 2, 0),
		nn.BatchNorm2d(self.out_channels),
		nn.Tanh(),
		)

	def forward(self, x):
		batch, time, z_dim = x.shape
		x = rearrange(x, 'b t z -> (b t) z')
		x = self.dec(x)
		x = rearrange(x, '(b t) c w h -> b t c w h', b=batch, t=time)
		return x

class EncoderConv(nn.Module):
	def __init__(self, channels, 
						h_dim):
		super(EncoderConv, self).__init__()
		self.h_dim = h_dim
		self.enc = nn.Sequential(
				nn.Conv2d(channels, 256, 5, 1, 2),         
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				nn.Conv2d(256, 256, 5, 2, 2),         
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				nn.Conv2d(256, 256, 5, 2, 2),          
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				nn.Conv2d(256, 256, 5, 2, 2),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				nn.Conv2d(256, 256, 5, 1, 2),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				Rearrange('b c w h -> b (c w h)'),
				nn.Linear(256*8*8, 4096),
				nn.BatchNorm1d(4096),
				nn.LeakyReLU(0.2),
				nn.Linear(4096, self.h_dim),
				nn.BatchNorm1d(self.h_dim),
				nn.LeakyReLU(0.2),
			)

	def forward(self, x):
		batch, time, channels, rows, width = x.shape
		x = rearrange(x, 'b t c r w -> (b t) c r w')
		x_t = self.enc(x)
		x_t = rearrange(x_t, '(b t) z -> b t z', b=batch, t=time)
		return x_t

class EncoderConv1(nn.Module):
	def __init__(self, channels, 
						h_dim):
		super(EncoderConv1, self).__init__()
		self.h_dim = h_dim
		self.enc = nn.Sequential(
				nn.Conv2d(channels, 32, 5, 1, 2),         
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2),
				nn.Conv2d(32, 64, 5, 2, 2),         
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2),
				nn.Conv2d(64, 128, 5, 2, 2),          
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2),
				nn.Conv2d(128, 256, 5, 2, 2),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				Rearrange('b c w h -> b (c w h)'),
				nn.Linear(256*4*4, self.h_dim),
				nn.BatchNorm1d(self.h_dim),
				nn.LeakyReLU(0.2),
			)

	def forward(self, x):
		batch, time, channels, rows, width = x.shape
		x = rearrange(x, 'b t c w h -> (b t) c w h')
		x_t = self.enc(x)
		x_t = rearrange(x_t, '(b t) z -> b t z', b=batch, t=time)
		return x_t

class EncoderConv2(nn.Module):
	def __init__(self, channels, 
					h_dim):
		super(EncoderConv2, self).__init__()
		self.h_dim = h_dim
		self.enc = nn.Sequential(
				nn.Conv2d(channels, 64, 4, 2, 1),         
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2),
				nn.Conv2d(64, 128, 4, 2, 1),     
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2),
				nn.Conv2d(128, 256, 4, 2, 2),    
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				nn.Conv2d(256, 512, 4, 2, 2),
				nn.BatchNorm2d(512),
				nn.LeakyReLU(0.2),
				nn.Conv2d(512, 128, 4, 1, 0),
				Rearrange('b c w h -> b (c w h)'),
				nn.Linear(128*2*2, self.h_dim),
				nn.BatchNorm1d(self.h_dim),
				nn.Tanh(),
			)

	def forward(self, x):
		batch, time, channels, rows, width = x.shape
		x = rearrange(x, 'b t c w h -> (b t) c w h')
		x_t = self.enc(x)
		x_t = rearrange(x_t, '(b t) z -> b t z', b=batch, t=time)
		return x_t


class EncoderResNet(nn.Module):
	def __init__(self, rows, 
				columns,
				channels, 
				h_dim):
		super(EncoderResNet, self).__init__()
		self.h_dim = h_dim
		self.enc = []
		self.enc.append(ResLayer2D(channels, 64, 3, norm=True, sample='down', activation='relu', channeldrop=False))
		self.enc.append(SelfAttention(64))
		self.enc.append(ResLayer2D(64, 128, 3, norm=True, sample='down', activation='relu', channeldrop=False)) 
		self.enc.append(ResLayer2D(128, 256, 3, norm=True,sample='down',  activation='relu', channeldrop=False)) 
		self.enc.append(ResLayer2D(256, 512, 3, norm=True, sample='down', activation='relu', channeldrop=False)) 
		self.enc.append(ResLayer2D(512, 1024, 3, norm=True,sample='down', activation='relu', channeldrop=False))

		self.enc = nn.ModuleList(self.enc)
		
		self.enc_fc = []
		self.enc_fc.append(nn.Sequential(
						Rearrange('b c w h -> b (c w h)'),
						nn.Linear(64*int(rows/2)*int(columns/2), h_dim),
						))

		self.enc_fc.append(nn.Sequential(
						Rearrange('b c w h -> b (c w h)'),
						nn.Linear(128*int(rows/4)*int(columns/4), h_dim),
						))
		self.enc_fc.append(nn.Sequential(
						Rearrange('b c w h -> b (c w h)'),
						nn.Linear(256*int(rows/8)*int(columns/8), h_dim),
						))
		self.enc_fc.append(nn.Sequential(
						Rearrange('b c w h -> b (c w h)'),
						nn.Linear(512*int(rows/16)*int(columns/16), h_dim),
						))
		self.enc_fc.append(nn.Sequential(
						Rearrange('b c w h -> b (c w h)'),
						nn.Linear(1024*int(rows/32)*int(columns/32), h_dim),
						))
		self.enc_fc.append(nn.Linear(self.h_dim, self.h_dim))
		self.enc_fc = nn.ModuleList(self.enc_fc)

		self.finallayer = nn.Sequential(
						nn.Linear(len(self.enc_fc)*self.h_dim, self.h_dim),
						nn.BatchNorm1d(self.h_dim),
						nn.ReLU(True),
						)

	def forward(self, x):
		batch, time, channels, rows, width = x.shape
		x = rearrange(x, 'b t c w h -> (b t) c w h')
		x_t = []
		i = 0
		for layer in range(len(self.enc)):
			x = self.enc[layer](x)
			if layer==1:
				continue
			h = self.enc_fc[i](x)
			i += 1
			x_t.append(h)
		x_t.append(self.enc_fc[-1](h))
		nm_ft = len(x_t)
		x_t = torch.stack(x_t, dim=-1)
		x_t = rearrange(x_t, 'b h n -> b (h n)')
		x_t = self.finallayer(x_t)
		x_t = rearrange(x_t, '(b t) h -> b t h', b=batch, t=time)
		return x_t



class DecoderConv(nn.Module):
	def __init__(self, h_dim, 
						v_dim,
						z_dim, 
						out_channels):
		super(DecoderConv, self).__init__()
		self.out_channels = out_channels
		in_dim = z_dim + v_dim
		self.dec = nn.Sequential(
				nn.Linear(in_dim, 2*h_dim),
				nn.BatchNorm1d(2*h_dim),
				nn.LeakyReLU(0.2),
				nn.Linear(2*h_dim, 256*8*8),
				nn.BatchNorm1d(256*8*8),
				nn.LeakyReLU(0.2),
				Rearrange('b (c w h) -> b c w h', c=256, w=8, h=8),
				nn.ConvTranspose2d(256, 256, 5, 2, 2, 1),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(256, 256, 5, 2, 2, 1),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(256, 256, 5, 2, 2, 1),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(256, self.out_channels, 5, 1, 2, 0),
				nn.BatchNorm2d(self.out_channels),
				nn.Tanh(),
			)

	def forward(self, x):
		batch, time, z_dim = x.shape
		x = rearrange(x, 'b t z -> (b t) z')
		x = self.dec(x)
		x = rearrange(x, '(b t) c w h -> b t c w h', b=batch, t=time)
		return x


class DecoderConv1(nn.Module):
	def __init__(self, h_dim, 
						v_dim,
						z_dim, 
						out_channels):
		super(DecoderConv1, self).__init__()
		self.out_channels = out_channels
		in_dim = z_dim + v_dim
		self.dec = nn.Sequential(
				nn.Linear(in_dim, h_dim),
				nn.BatchNorm1d(h_dim),
				nn.LeakyReLU(0.2),
				nn.Linear(h_dim, 2048),
				nn.BatchNorm1d(2048),
				nn.LeakyReLU(0.2),
				Rearrange('b (c w h) -> b c w h', c=128, w=4, h=4),
				nn.ConvTranspose2d(128, 256, 5, 2, 2, 1),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(256, 128, 5, 2, 2, 1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(128, 64, 5, 2, 2, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2),
				nn.ConvTranspose2d(64, 3, 5, 1, 2, 0),
				nn.BatchNorm2d(3),
				nn.Tanh(),
			)

	def forward(self, x):
		batch, time, z_dim = x.shape
		x = rearrange(x, 'b t z -> (b t) z')
		x = self.dec(x)
		x = rearrange(x, '(b t) c w h-> b t c w h', b=batch, t=time)
		return x


class DecoderConv2(nn.Module):
	def __init__(self, h_dim, 
						v_dim,
						z_dim,
						out_channels):
		super(DecoderConv2, self).__init__()
		self.out_channels = out_channels
		in_dim = z_dim + v_dim
		self.dec = nn.Sequential(
				nn.Linear(in_dim, h_dim),
				nn.BatchNorm1d(h_dim),
				nn.ReLU(),
				nn.Linear(h_dim, 3200),
				nn.BatchNorm1d(3200),
				nn.ReLU(),
				Rearrange('b (c w h) -> b c w h', c=128, w=5, h=5),
				nn.ConvTranspose2d(128, 512, 4, 1, 0, 0),
				nn.BatchNorm2d(512),
				nn.ReLU(),
				nn.Upsample(scale_factor=(2,2)),
				nn.Conv2d(512, 256, 3, 1, 1),
				nn.BatchNorm2d(256),
				nn.ReLU(),
				nn.Upsample(scale_factor=(2,2)),
				nn.Conv2d(256, 128, 3, 1, 1),
				nn.BatchNorm2d(128),
				nn.ReLU(),
				nn.Upsample(scale_factor=(2,2)),
				nn.Conv2d(128, 64, 3, 1, 1),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.Conv2d(64, 3, 1, 1, 0),
				nn.Sigmoid(),
			)

	def forward(self, x):
		batch, time, z_dim = x.shape
		x = x.reshape(-1, z_dim)
		x = self.dec(x)
		x = rearrange(x, '(b t) c w h-> b t c w h', b=batch, t=time)
		return x


class DecoderResNet(nn.Module):
	def __init__(self, rows,
					columns,
					channels,
					h_dim, 
					v_dim, 
					z_dim):
		super(DecoderResNet, self).__init__()
		self.channels = channels
		in_dim = v_dim + z_dim

		self.linear = nn.Linear(in_dim, h_dim)
		self.dec_fc = []

		self.dec_fc.append(nn.Sequential(
						nn.Linear(h_dim, ht_dim*int(rows/32)*int(columns/32)),
						Rearrange('b (c w h) -> b c w h', c=ht_dim, w=int(rows/32), h=int(columns/32)),
						))

		self.dec_fc.append(nn.Sequential(
						nn.Linear(h_dim, 1024*int(rows/32)*int(columns/32)),
						Rearrange('b (c w h) -> b c w h', c=1024, w=int(rows/32), h=int(columns/32)),
						))

		self.dec_fc.append(nn.Sequential(
						nn.Linear(h_dim, 512*int(rows/16)*int(columns/16)),
						Rearrange('b (c w h) -> b c w h', c=512, w=int(rows/16), h=int(columns/16)),
						))

		self.dec_fc.append(nn.Sequential(
						nn.Linear(h_dim, 256*int(rows/8)*int(columns/8)),
						Rearrange('b (c w h) -> b c w h', c=256, w=int(rows/8), h=int(columns/8)),
						))

		self.dec_fc.append(nn.Sequential(
						nn.Linear(h_dim, 128*int(rows/4)*int(columns/4)),
						Rearrange('b (c w h) -> b c w h', c=128, w=int(rows/4), h=int(columns/4)),
						))

		self.dec_fc.append(nn.Sequential(
						nn.Linear(h_dim, 64*int(rows/2)*int(columns/2)),
						Rearrange('b (c w h) -> b c w h', c=64, w=int(rows/2), h=int(columns/2)),
						))


		self.dec_fc = nn.ModuleList(self.dec_fc)

		self.dec = []
		self.dec.append(ResLayer2D(3072, 1024, 3, norm=True, sample='up', activation='relu', channeldrop=True))
		self.dec.append(ResLayer2D(1536, 512, 3, norm=True, sample='up', activation='relu', channeldrop=True)) 
		self.dec.append(ResLayer2D(768, 256, 3, norm=True,sample='up',  activation='relu', channeldrop=True)) 
		self.dec.append(ResLayer2D(384, 128, 3, norm=True, sample='up', activation='relu', channeldrop=True)) 
		self.dec.append(SelfAttention(192))
		self.dec.append(ResLayer2D(192, 64, 3, norm=True,sample='up', activation='relu', channeldrop=True))
		self.dec.append(nn.Sequential(
					nn.Conv2d(64, 3, 3, 1, 1),
					nn.Sigmoid(),
				))

		self.dec = nn.ModuleList(self.dec)

	def forward(self, z):
		batch, time, z_dim = z.shape
		z = rearrange(z, 'b t z -> (b t) z')
		z = self.linear(z)
		h = torch.cat([self.dec_fc[0](z), self.dec_fc[1](z)], 1)
		h = self.dec[0](h)
		for layer in range(len(self.dec)-3):
			h1 = self.dec_fc[layer+2](z)
			h = torch.cat([h, h1], 1)
			h = self.dec[layer+1](h)
		h = self.dec[-2](h)
		h = self.dec[-1](h)
		h = rearrange(h, '(b t) c w h -> b t c w h', b=batch, t=time)
		return h


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, x):
        max_vals, _ = torch.max(torch.abs(x), 2, keepdim=True)
        max_vals  = max_vals + 1e-5
        x = x / max_vals
        return x
    
from torch.nn.utils import weight_norm
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                    self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)



class DynamicalModel(nn.Module):
	def __init__(self, v_dim, 
				nm_operators,  
				dynamics='Hamiltonian',
				condnU=True,
				projection=True,
				block_size=2):	
		super(DynamicalModel, self).__init__()
		self.dynamics = dynamics
		self.nm_operators = nm_operators
		self.condnU = condnU

		self.projection = projection

		self.subspace_dim = int(v_dim/nm_operators)

		if self.dynamics == 'symplecticform':
			self.H = nn.Parameter(torch.Tensor(self.nm_operators, self.subspace_dim, self.subspace_dim), requires_grad=True)
			init.orthogonal_(self.H.data, gain=np.sqrt(2))

		elif self.dynamics == 'skewsymHamiltonian':
			self.H = nn.Parameter(torch.Tensor(self.nm_operators, self.subspace_dim, self.subspace_dim), requires_grad=True)
			init.orthogonal_(self.H.data, gain=np.sqrt(2))

		elif self.dynamics == 'Hamiltonian':
			self.H = nn.Parameter(torch.Tensor(self.nm_operators, 2*self.subspace_dim, 2*self.subspace_dim), requires_grad=True)
			init.orthogonal_(self.H.data, gain=np.sqrt(2))

		elif self.dynamics in ['RNN', 'Linear']:
			if self.condnU:
				self.W_ih = nn.Parameter(torch.Tensor(self.nm_operators, self.subspace_dim, self.subspace_dim))
				self.W_hh = nn.Parameter(torch.Tensor(self.nm_operators, self.subspace_dim, self.subspace_dim))
				self.b = nn.Parameter(torch.Tensor(self.nm_operators, self.subspace_dim))
			else:
				self.W_ih = nn.Parameter(torch.Tensor(self.nm_operators*self.subspace_dim, self.nm_operators*self.subspace_dim))
				self.W_hh = nn.Parameter(torch.Tensor(self.nm_operators*self.subspace_dim, self.nm_operators*self.subspace_dim))
				self.b = nn.Parameter(torch.Tensor(self.nm_operators*self.subspace_dim))
				
			init.orthogonal_(self.W_ih.data, gain=np.sqrt(2))
			init.orthogonal_(self.W_hh.data, gain=np.sqrt(2))
			init.constant_(self.b.data, 0.0)
		else:
			assert 0,f"Not implemented {self.dynamics}"


	def spectral_norm(self, W):
		return W/torch.linalg.norm(W, ord=2, dim=(1,2)).unsqueeze(1).unsqueeze(2)


	def hamiltonian(self, device='cpu'):
		if self.dynamics == 'Hamiltonian':
			A = (self.H + self.H.transpose(2,1)) * 0.5
			J = torch.eye(self.subspace_dim).to(device)
			J = repeat(J, 'i j -> n i j', n=self.nm_operators)
			H = torch.zeros(self.nm_operators, 2*self.subspace_dim, 2*self.subspace_dim).to(device)
			H[:, :self.subspace_dim,self.subspace_dim:] = J
			H[:, self.subspace_dim:,:self.subspace_dim] = -J
			H = torch.einsum('n i k, n k j -> n i j', H, A)

		elif self.dynamics == 'skewsymHamiltonian':
			A = (self.H + self.H.transpose(2,1)) * 0.5
			H = torch.zeros(self.nm_operators, 2*self.subspace_dim, 2*self.subspace_dim).to(device)
			H[:, :self.subspace_dim, self.subspace_dim:] = A
			H[:, self.subspace_dim:, :self.subspace_dim] = -A

		else:
			assert 0,f"Not implemented {self.dynamics}"

		return H

	def hamiltonianForm(self, s_t, start, timesteps):
		H = self.hamiltonian(s_t.device)
		expH = self.matrix_exp(H)
		if start>0:
			backexpH = self.matrix_exp(-H)
		s_t_seq = [s_t]
		for t in range(start, timesteps-1):
			s_t_seq.append(torch.einsum('kid, bkd -> bki', expH, s_t_seq[-1]))
		for t in range(0, start):
			s_t_seq.insert(0, torch.einsum('kid, bkd -> bki', backexpH, s_t_seq[0]))
		s_t_seq = torch.stack(s_t_seq, 1)
		return s_t_seq
	
	def rnn_or_linear(self, s_t, timesteps):
		samples = s_t.shape[0]
		if self.condnU:
			h0 = torch.zeros(samples, self.nm_operators, self.subspace_dim).to(s_t.device)
			s_t_seq, h = [], [h0]
			for t in range(timesteps):
				sh = torch.einsum('kid, bkd -> bki', self.W_ih, s_t[:,t,:,:])
				hh = torch.einsum('kid, bkd -> bki', self.W_hh, h[-1])
				s_next = torch.einsum('bki, bki, ki -> bki', sh, hh, self.b)
				if self.dynamics == 'RNN':
					s_next = torch.tanh(s_next)
				s_t_seq.append(s_next)
				h.append(s_next)
			s_t_seq = torch.stack(s_t_seq, 1)
		else:
			s_t = rearrange(s_t, 'b t n d -> b t (n d)')
			s_t = s_t/(s_t.norm(dim=2).unsqueeze(2) + 1e-8)
			h0 = torch.zeros(samples, self.subspace_dim*self.nm_operators).to(s_t.device)
			s_t_seq, h = [], [h0]
			for t in range(timesteps):
				sh = torch.einsum('id, bd -> bi', self.W_ih, s_t[:,t,:])
				hh = torch.einsum('id, bd -> bi', self.W_hh, h[-1])
				s_next = torch.einsum('bi, bi, i -> bi', sh, hh, self.b)
				if self.dynamics == 'RNN':
					s_next = torch.tanh(s_next)
				s_t_seq.append(s_next)
				h.append(s_next)
			s_t_seq = torch.stack(s_t_seq, 1)
			s_t_seq = repeat(s_t_seq, 'b t d -> b t n d', n=1)
		return s_t_seq

	def test_rnn_or_linear(self, s_t, timesteps):
		samples = s_t.shape[0]
		if self.condnU:
			h0 = torch.zeros(samples, self.nm_operators, self.subspace_dim).to(s_t.device)
			s_t_seq, h = [s_t], [h0]
			for t in range(timesteps-1):
				if t==0:
					sh = torch.einsum('kid, bkd -> bki', self.W_ih, s_t)
				else:
					sh = torch.zeros(*h0.shape).to(s_t.device)
				hh = torch.einsum('kid, bkd -> bki', self.W_hh, h[-1])
				s_next = torch.einsum('bki, bki, ki -> bki', sh, hh, self.b)
				if self.dynamics == 'RNN':
					s_next = torch.tanh(s_next)
				s_t_seq.append(s_next)
				h.append(s_next)
			s_t_seq = torch.stack(s_t_seq, 1)
		else:
			s_t = rearrange(s_t, 'b n d -> b (n d)')
			s_t = s_t/(s_t.norm(dim=1).unsqueeze(1) +1e-8)
			h0 = torch.zeros(samples, self.subspace_dim*self.nm_operators).to(s_t.device)
			s_t_seq, h = [s_t], [h0]
			for t in range(timesteps-1):
				if t==0:
					sh = torch.einsum('id, bd -> bi', self.W_ih, s_t)
				else:
					sh = torch.zeros(*h0.shape).to(s_t.device)
				#sh = torch.einsum('id, bd -> bi', self.W_ih, s_t[:,t,:])
				hh = torch.einsum('id, bd -> bi', self.W_hh, h[-1])
				s_next = torch.einsum('bi, bi, i -> bi', sh, hh, self.b)
				if self.dynamics == 'RNN':
					s_next = torch.tanh(s_next)
				s_t_seq.append(s_next)
				h.append(s_next)
			s_t_seq = torch.stack(s_t_seq, 1)
			s_t_seq = repeat(s_t_seq, 'b t d -> b t n d', n=1)
		return s_t_seq
	


	def forward(self, u_t, s_t=[], timesteps=8, startidx=0, test=False):
		samples = u_t.shape[0]
		labels = u_t.argmax(1)
		if len(s_t) == 0:
			if self.dynamics == 'Hamiltonian':
				s_t = torch.randn(samples, self.nm_operators, 2*self.subspace_dim).to(u_t.device)
			else:
				s_t = torch.randn(samples, self.nm_operators, self.subspace_dim).to(u_t.device)

		if not self.projection:
			s_mask = repeat(s_t, 'b n v-> b t n v', t=timesteps)

		elif self.dynamics in ['skewsymHamiltonian', 'Hamiltonian']:
			s_t_seq = self.hamiltonianForm(s_t, startidx, timesteps)
		elif self.dynamics in ['RNN', 'Linear']:
			if test:
				s_t_seq = self.test_rnn_or_linear(s_t, timesteps)
			else:
				s_t_seq = self.rnn_or_linear(s_t, timesteps)
		else:
			assert 0, f"Not Implemented {self.dynamics}"

		if self.condnU:
			if self.projection:
				projection = torch.zeros(*s_t_seq.shape).to(s_t_seq.device)
				projection[range(len(labels)),:,labels,:] = 1.
				s_t_seq = s_t_seq * projection
			else:
				s_mask[range(len(labels)),:,labels,:] = s_t_seq.clone()[range(len(labels)),:,labels,:]
				s_t_seq = s_mask.clone()
		return s_t_seq

	def sampling(self, u_t, timesteps=8, startidx=0):
		samples = u_t.shape[0]
		labels = u_t.argmax(1)
		if self.dynamics == 'Hamiltonian':
			dim = 2*self.subspace_dim
		else:
			dim = self.subspace_dim
		s_t = torch.randn(samples, self.nm_operators, dim).to(u_t.device)
		s_seq = self.forward(u_t, s_t, timesteps, startidx, test=True)
		return s_seq


	def matrix_exp(self, x, order=10, usetorch=False):
		if usetorch:
			return torch.matrix_exp(x)
		else:
			I = torch.eye(x.size(0))
			out = I
			power = I
			for i in range(order):
				power = torch.matmul(power, x) / (i+1)
				out += power
			return out


