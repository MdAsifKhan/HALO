import os
import sys
import subprocess
import shutil
import pdb
import glob
import re
import cv2

class PreprocessMUG:
	def __init__(self, dataroot, nm_frames=8, downsample=4, stride=8):
		self.dataroot = dataroot
		self.nm_frames = nm_frames
		self.downsample = downsample
		self.stride = stride
		self.cascade_path = '/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
		self.actions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

		if not os.path.isdir(f"{self.dataroot}/TRAIN"):
			os.makedirs(f"{self.dataroot}/TRAIN" )
		if not os.path.isdir(f"{self.dataroot}/VALIDATION"):
			os.makedirs(f"{self.dataroot}/VALIDATION")

		self.subjects = os.listdir(f"{dataroot}/raw/")
		nm_train = int(0.8*len(self.subjects))
		self.train = self.subjects[:nm_train]
		self.test = self.subjects[nm_train:]
		self.frame_regex = re.compile(r'([0-9]+).jpg')
		self.create_data()
	
	def frame_idx(self, idx):
		return re.search(self.frame_regex, idx).group(1)

	def detect_face(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		cascade = cv2.CascadeClassifier(self.cascade_path)
		rect = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(96,96))
		rect = rect[0] if len(rect)>0 else None
		return rect

	def save_sequence(self, action_path, subject, session, action):
		if subject in self.train:
			output_path = f"{self.dataroot}/TRAIN/Person{subject}_{session}_Action_{action}"
		else:
			output_path = f"{self.dataroot}/VALIDATION/Person{subject}_{session}_Action_{action}"
		seq = []
		images  = glob.glob(os.path.join(action_path, '*.jpg'))
		if len(images) == 0:
			subfolder = os.listdir(action_path)
			if len(subfolder)!=0:
				for take in subfolder:
					images = glob.glob(os.path.join(action_path, take, '*.jpg'))
					if len(images)<40:
						continue
					else:
						images = sorted(images, key=self.frame_idx)
						seq.append(images)
		elif len(images)>0 and len(images)<40:
			pass
		else:
			images  = sorted(images, key=self.frame_idx)
		nm_seq = 0
		for i,sq in enumerate(seq):
			start, end = 0, len(sq)-int(0.1*len(sq))
			for idx, offset  in enumerate(range(start, end-self.downsample*self.nm_frames, self.stride)):
				folder_name = f"{output_path}_take_{i}_sequence_{idx}_length_{self.nm_frames}"
				os.makedirs(folder_name, exist_ok=True)
				
				image = cv2.imread(sq[offset + (self.downsample*self.nm_frames)//2])
				box = self.detect_face(image)
				if box is None:
					shutil.rmtree(folder_name)
					continue
				for frame in range(self.nm_frames):
					image_f = cv2.imread(sq[offset + self.downsample*frame])
					x, y, w, h = box[0], box[1], box[2], box[3]
					image_f = image_f[y:y+h, x:x+w]
					image_f = cv2.resize(image_f, (64,64))
					cv2.imwrite(f"{folder_name}/{frame+1:02d}.jpg", image_f)
			nm_seq += 1
		print(f"Number of Sequence {nm_seq} for Subject {subject} Action {action} Session {session}")
		return nm_seq

	def create_data(self):
		total_seq = 0
		for subject in self.subjects:
			subject_path = f"{self.dataroot}/raw/{subject}"
			for session_n in os.listdir(subject_path):
				if 'session' in session_n:
					subject_actions = f"{subject_path}/{session_n}"
					for sub_action in os.listdir(subject_actions):
						if sub_action in self.actions and os.path.isdir(subject_actions):
							action_path = f"{subject_path}/{session_n}/{sub_action}/"
							nm_seq = self.save_sequence(action_path, subject, session_n, sub_action)
				else:
					sub_action = session_n
					if sub_action in self.actions:
						action_path = f"{self.dataroot}/raw/{subject}/{sub_action}/"
						nm_seq = self.save_sequence(action_path, subject, 'session0', sub_action)
				total_seq += nm_seq
		print(f"Total Sequence {total_seq}")


import sys
if __name__ == '__main__':
	if len(sys.argv)==0:
		assert 0, "Specify data path"
	else:
		dataroot = sys.argv[1]
		dataname = sys.argv[2]
		os.path.isdir(dataroot)
	if dataname == 'MUG':
		preprocess = PreprocessMUG(dataroot)
	else:
		assert 0, f"Not implemented {preprocess}"
