import cv2
import pudb
from random import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

TRANSFER = 25
TOTAL = 300

f = open("pic_list/['Smiling', 'Male']~[]", "r")
pics = []

while True:
	line = f.readline()
	if not line: break
	pics.append(line.strip())

f.close()

idx_1 = [int(random() * len(pics)) for x in range(TOTAL)]

org_img_list = [pics[x] for x in idx_1]

gen_img_list = ["pictures/smile_1000/male_smiling_"+str(x)+".png" for x in range(TOTAL)]

all_pic_matrices = {}

for pic in org_img_list:
	all_pic_matrices[pic] = cv2.resize(cv2.imread(pic), (64, 64))

for pic in gen_img_list:
	all_pic_matrices[pic] = cv2.imread(pic)
	

klds = []

for it in tqdm(range(100)):

	
	imgs_1 = [all_pic_matrices[pic] for pic in org_img_list]
	imgs_2 = [all_pic_matrices[pic] for pic in gen_img_list]

	imgs_1 = torch.tensor(imgs_1).flatten()
	imgs_2 = torch.tensor(imgs_2).flatten()

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

	klds.append(kld)

	if it == 0: base_kld = kld
	
	idx_1 = [int(random() * len(org_img_list)) for x in range(TRANSFER)]
	idx_2 = [int(random() * len(gen_img_list)) for x in range(TRANSFER)]

	temp = [org_img_list[x] for x in idx_1]
	
	for i, idx in enumerate(idx_1):
		org_img_list[idx] = gen_img_list[idx_2[i]]

	for i, idx in enumerate(idx_2):
		gen_img_list[idx] = temp[i]

plt.hist(klds)
plt.savefig("hist_smile.png")

print("Base kld: "+str(base_kld))
pu.db