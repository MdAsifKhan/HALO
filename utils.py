import matplotlib.pyplot as plt

from celluloid import Camera
from matplotlib import cm, gridspec
import numpy as np
import pdb
from matplotlib.animation import FuncAnimation, writers
import os
import pdb

def atanh(x):
	return 0.5*torch.log((1+x)/(1-x))

def makedir(folder):
	if not os.path.isdir(folder):
		os.makedirs(folder)

def plot_latent_2d(z_t, z_tp1, out_path, label=[]):
	plt.subplot(2,1,1)
	for seq in range(z_t.shape[0]):
		if len(label)>0:
			plt.plot(z_t[seq,:,0], z_t[seq,:,1], 'o--', label=label[seq])
		else:
			plt.plot(z_t[seq,:,0], z_t[seq,:,1], 'o--', label=f"orig_{seq}")
	plt.subplot(2,1,2)
	for seq in range(z_tp1.shape[0]):
		if len(label)>0:
			plt.plot(z_tp1[seq,:,0], z_tp1[seq,:,1], 'o--', label=label[seq])
		else:		
			plt.plot(z_tp1[seq,:,0], z_tp1[seq,:,1], 'o--', label=f"pred_{seq}")
	plt.savefig(out_path)
	plt.legend(loc='lower right')
	plt.cla()
	plt.clf()
	plt.close()


def plot_latent_3d(z_t, z_tp1, out_path, label=[]):
	fig = plt.figure()
	ax = fig.add_subplot(211, projection='3d')
	for seq in range(z_t.shape[0]):
		if len(label)>0:
			ax.plot(z_t[seq,:,0], z_t[seq,:,1],  z_t[seq,:,2], 'o--', label=label[seq])
		else:
			ax.plot(z_t[seq,:,0], z_t[seq,:,1],  z_t[seq,:,2], 'o--', label=f"orig_{seq}")
	ax = fig.add_subplot(212, projection='3d')
	for seq in range(z_tp1.shape[0]):
		if len(label)>0:
			ax.plot(z_t[seq,:,0], z_t[seq,:,1],  z_t[seq,:,2], 'o--', label=label[seq])
		else:
			ax.plot(z_tp1[seq,:,0], z_tp1[seq,:,1],  z_tp1[seq,:,2], 'o--', label=f"pred_{seq}")
	plt.savefig(out_path)
	plt.legend(loc='lower right')
	plt.cla()
	plt.clf()
	plt.close()


def save_seq_gif(seq, out_path, fps=20):
	ncols = int(np.ceil(seq.shape[0]**0.5))
	nrows = 1 + seq.shape[0]//ncols
	fig = plt.figure()
	camera = Camera(fig)
	for k in range(seq.shape[1]):
		for i in range(seq.shape[0]):
			ax = fig.add_subplot(ncols, nrows, i+1)
			ax.imshow(seq[i,k])
			ax.set_axis_off()
		camera.snap()
	anim = camera.animate()
	anim.save(f"{out_path}", writer='imagemagick', fps=fps)
	plt.cla()
	plt.clf()
	plt.close()

def save_seq_img(seq, out_path):
	ncols = seq.shape[0]
	nrows = seq.shape[1]
	ch = seq.shape[-1]
	fig = plt.figure(figsize=(nrows+1, ncols+1))
	gs = gridspec.GridSpec(ncols, nrows)
	gs.update(wspace=0, hspace=0, top=1-0.5/(ncols+1), bottom=0.5/(ncols+1), left=0.5/(nrows+1), right=1-0.5/(nrows+1))
	c = 0
	for i in range(seq.shape[0]):
		for k in range(seq.shape[1]):
			#fig.add_subplot(nrows, ncols, c+1)
			ax = fig.add_subplot(gs[c])
			#ax = plt.subplots(gs[i,k])
			if ch == 1:
				ax.imshow(seq[i,k].squeeze(), cmap='gray')
			else:
				ax.imshow(seq[i,k])
			ax.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			c += 1
	#gs.tight_layout(fig)
	#gs.tight_layout(fig)
	plt.savefig(f"{out_path}")
	plt.cla()
	plt.clf()
	plt.close()		



def save_latent_zandzp1(z, zp1, out_path):
	colors = cm.rainbow(np.linspace(0, 1, len(z)))
	fig, ax = plt.subplots(2, sharex=True)
	camera = Camera(fig)
	for j in range(z.shape[0]):
		for i in range(z.shape[1]):
			ax[0].plot(z[j,:i,0], z[j,:i,1], 'o--', c=colors[j])
			ax[1].plot(zp1[j,:i,0], zp1[j,:i,1], 'o--', c=colors[j])
			for k in range(i):
				ax[0].annotate(str(k+1), (z[j,k,0], z[j,k,1]), fontsize=6)
				ax[1].annotate(str(k+1), (zp1[j,k,0], zp1[j,k,1]), fontsize=6)
			camera.snap()
	anim = camera.animate()
	anim.save(f"{out_path}seq.gif", writer='imagemagick', fps=3)
	plt.cla()
	plt.clf()
	plt.close()

def save_latent_z(z, out_path):
	colors = cm.rainbow(np.linspace(0, 1, len(z)))
	fig = plt.figure()
	camera = Camera(fig)
	for j in range(z.shape[0]):
		for i in range(z.shape[1]):
			plt.plot(z[j,:i,0], z[j,:i,1], 'o--', c=colors[j])
			for k in range(i):
				plt.annotate(str(k+1), (z[j,k,0], z[j,k,1]), fontsize=6)
			camera.snap()
	anim = camera.animate()
	anim.save(f"{out_path}seq.gif", writer='imagemagick', fps=3)
	plt.cla()
	plt.clf()
	plt.close()


def save_latent_zandzp13D(z, zp1, out_path):
	colors = cm.rainbow(np.linspace(0, 1, len(z)))
	fig, ax = plt.subplots(2, sharex=True)
	camera = Camera(fig)
	for j in range(z.shape[0]):
		for i in range(z.shape[1]):
			ax[0].plot(z[j,:i,0], z[j,:i,1], 'o--', c=colors[j])
			ax[1].plot(zp1[j,:i,0], zp1[j,:i,1], 'o--', c=colors[j])
			for k in range(i):
				ax[0].annotate(str(k+1), (z[j,k,0], z[j,k,1]), fontsize=6)
				ax[1].annotate(str(k+1), (zp1[j,k,0], zp1[j,k,1]), fontsize=6)
			camera.snap()
	anim = camera.animate()
	anim.save(f"{out_path}seq.gif", writer='imagemagick', fps=3)
	plt.cla()
	plt.clf()
	plt.close()

def save_latent_z3D(z, out_path):
	colors = cm.rainbow(np.linspace(0, 1, len(z)))
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	camera = Camera(fig)
	for j in range(z.shape[0]):
		for i in range(z.shape[1]):
			ax.plot(z[j,:i,0], z[j,:i,1], z[j,:i,2], 'o--', c=colors[j])
			#for k in range(i):
			#	ax.annotate(str(k+1), (z[j,k,0], z[j,k,1]), fontsize=6)
			camera.snap()
		#plt.title('seq{}'.format(j+1))
	anim = camera.animate()
	#plt.legend(loc='lower right')
	anim.save(f"{out_path}latent_seq.gif", writer='imagemagick', fps=3)
	plt.cla()
	plt.clf()
	plt.close()


def latent_scatter_3d(x, batch_key_test, out_path):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i, (seq, label) in enumerate(zip(x, batch_key_test)):
		ax.plot(seq[:,0], seq[:,1], seq[:,2], 'o--', label=label)
		#for k in range(x.shape[1]):
		#	ax.text(seq[k,0], seq[k,1], seq[k,2], str(k+1), fontsize=6)
	#ax.set_xlim([-1, 1])
	#ax.set_ylim([-1, 1])
	#ax.set_zlim([-1, 1])
	ax.legend(fontsize=6)
	plt.savefig(f"{out_path}.png")
	plt.cla()
	plt.clf()
	plt.close()

def latent_scatter_2d(x, batch_key_test, out_path):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i, (seq, label) in enumerate(zip(x, batch_key_test)):
		ax.plot(seq[:,0], seq[:,1], 'o--', label=label)
		#for k in range(x.shape[1]):
		#	ax.text(seq[k,0], seq[k,1], str(k+1), fontsize=6)
	ax.legend(fontsize=6)
	plt.savefig(f"{out_path}2d.png")
	plt.cla()
	plt.clf()
	plt.close()
	
#from mocap import viz
def save_animation(xyz_gt, out_path):
	# === Plot and animate ===
	fig = plt.figure()
	fig.tight_layout()
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	ob = viz.Ax3DPose(ax)
	def update(i):
		ob.update(xyz_gt[i,:], lcolor="#9b59b6", rcolor="#2ecc71")

	anim = FuncAnimation(fig, update, frames=xyz_gt.shape[0], interval=100, repeat=False)
	anim.save(f"{out_path}.gif", dpi=80, writer='imagemagick')
	#Writer = writers['ffmpeg']
	#writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
	#anim.save('{}.mp4'.format(out_path), writer=writer)
	plt.close()

def video_transform(video, image_transform):
	"""
	perform the image transformation and stack the video
	:param video: ndarray, with size [t, c, h, w]
	:param image_transform: the list of image transformation
	:return: video, tensor with size [c, t, h, w]
	"""
	vid = []
	for im in video:
	    vid.append(image_transform(im))

	vid = torch.stack(vid).permute(1, 0, 2, 3)
	return vid