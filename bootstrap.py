import cv2
import pudb
from random import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

f = open("list_attr_celeba.txt", "r")
f.readline()
f.readline()
pics = []

while True:
	line = f.readline()
	if not line: break
	pics.append(line.strip().split()[0])

f.close()

klds = []

for i in tqdm(range(100)):
	idx_1 = [int(random() * len(pics)) for x in range(100)]
	idx_2 = [int(random() * len(pics)) for x in range(100)]

	imgs_1, imgs_2 = [], []

	for j in idx_1:
		img_here = cv2.imread("img_align_celeba/"+pics[j])
		imgs_1.append(img_here)

	for j in idx_2:
		img_here = cv2.imread("img_align_celeba/"+pics[j])
		imgs_2.append(img_here)

	imgs_1 = torch.tensor(imgs_1).flatten()
	imgs_2 = torch.tensor(imgs_2).flatten()

	# a = torch.zeros((256,))
	# b = torch.zeros((256,))

	a = []
	b = []
	for i in range(256):
	    a.append((imgs_1 == i).unsqueeze(0))
	    b.append((imgs_2 == i).unsqueeze(0))

	a = torch.cat(a,dim=0).sum(dim=1).float()
	b = torch.cat(b,dim=0).sum(dim=1).float()


	a /= len(imgs_1)
	b /= len(imgs_2)

	kld = (a * (a / b).log()).sum()


	# out = F.kl_div(a, b)
	# pu.db

	# print("KL divergence: "+str(out))

	klds.append(kld)

plt.hist(klds)
plt.savefig("hist_all.png")
pu.db