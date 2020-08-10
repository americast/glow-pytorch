"""Train script.

Usage:
    infer_celeba.py <hparams> <dataset_root> <z_dir>
"""
import os
import cv2
import random
import torch
import vision
import numpy as np
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.config import JsonConfig
from glow.utils import load
from tqdm import tqdm
import pudb
from random import random
import pickle
import torch.nn.functional as F
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def save_images(images, names):
    if not os.path.exists("pictures/infer/"):
        os.makedirs("pictures/infer/")
    for img, name in zip(images, names):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite("dps/{}.png".format(name), img)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        print("Saved as dps/{}.png".format(name))

def run_z(graph, z):
    graph.eval()
    x = graph(z=torch.tensor([z]), eps_std=0.3, reverse=True)
    img = x[0].permute(1, 2, 0).detach().cpu().numpy()
    img = img[:, :, ::-1]
    img = cv2.resize(img, (64, 64))
    return img

def get_base_indices(filename):
    base_indices = []
    f = open("pic_list/"+filename, "r")
    while True:
        line = f.readline()
        if not line: break
        base_indices.append(int(line.strip().split("/")[1].split(".")[0]) - 1)
    f.close()
    return base_indices

def get_n_indices(base_indices, n):
    indices_here = []
    for i in range(n):
        num = int(random() * len(base_indices))
        while base_indices[num] in indices_here:
            num = int(random() * len(base_indices))
        indices_here.append(base_indices[num])
    return indices_here

def KL(P,Q):
     """ Epsilon is used here to avoid conditional code for
     checking that neither P nor Q is equal to 0. """
     epsilon = 0.00001

     # You may want to instead make copies to avoid changing the np arrays.
     # P = P+epsilon
     # Q = Q+epsilon

     divergence = np.sum(P*np.log(P/Q))
     return divergence

class classifier_data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, male_neutral, female_neutral, male_smiling, female_smiling, transform=None, val=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.idx = range(int(0.7*len(male_smiling)))
        if val:
            self.idx = len(male_smiling) - self.idx

        self.idx_all = []

        for i in self.idx:
            self.idx_all.append((i, 0))
            self.idx_all.append((i, 1))
            self.idx_all.append((i, 2))
            self.idx_all.append((i, 3))

        self.male_neutral, self.female_neutral, self.male_smiling, self.female_smiling = male_neutral, female_neutral, male_smiling, female_smiling


    def __len__(self):
        return len(self.idx_all)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        n, r = self.idx_all[idx]

        all_here = self.male_neutral[:n], self.female_neutral[:n], self.male_smiling[:n], self.female_smiling[:n]

        imgs_all = torch.zeros(len(self.idx_all), 3, 64, 64)
        for i, each in enumerate(all_here[r]):

            img = Image.open(each).convert("RGB")

            if self.transform:
                img = self.transform(img)

            imgs_all[i,:,:,:] = img

        return imgs_all, n, r

if __name__ == "__main__":
    

    f = open("pic_list/['Male']~['Smiling']", "r")
    male_neutral = []
    while True:
        line = f.readline()
        if not line: break
        male_neutral.append(line.strip())
    f.close()

    f = open("pic_list/['Smiling']~['Male']", "r")
    female_smiling = []
    while True:
        line = f.readline()
        if not line: break
        female_smiling.append(line.strip())
    f.close()

    f = open("pic_list/[]~['Smiling', 'Male']", "r")
    female_neutral = []
    while True:
        line = f.readline()
        if not line: break
        female_neutral.append(line.strip())
    f.close()

    male_smiling = ["pictures/smile_1000/male_smiling_"+str(x)+".png" for x in range(1000)]

    hparams = JsonConfig("hparams/celeba.json")
    # set transform of dataset
    transform = transforms.Compose([
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize(hparams.Data.resize),
        transforms.ToTensor()])

    transformed_dataset = classifier_data(male_neutral, female_neutral, male_smiling, female_smiling, transform=transform)

    dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=1)

    for data, n, r in dataloader:
        break

    pu.db





    """
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    z_dir = args["<z_dir>"]
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    if not os.path.exists(z_dir):
        print("Generate Z to {}".format(z_dir))
        os.makedirs(z_dir)
        generate_z = True
    else:
        print("Load Z from {}".format(z_dir))
        generate_z = False

    hparams = JsonConfig("hparams/celeba.json")
    dataset = vision.Datasets["celeba"]
    # set transform of dataset
    transform = transforms.Compose([
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize(hparams.Data.resize),
        transforms.ToTensor()])
    # build
    built = build(hparams, True)
    load('trained.pkg', built['graph'], device=torch.device('cpu'))
    graph = built['graph']
    dataset = dataset(dataset_root, transform=transform)

    # get Z
    if not generate_z:
        # try to load
        try:
            delta_Z = []
            for i in range(hparams.Glow.y_classes):
                z = np.load(os.path.join(z_dir, "detla_z_{}.npy".format(i)))
                delta_Z.append(z)
        except FileNotFoundError:
            # need to generate
            generate_z = True
            print("Failed to load {} Z".format(hparams.Glow.y_classes))
            quit()
    if generate_z:
        delta_Z = graph.generate_attr_deltaz(dataset)
        for i, z in enumerate(delta_Z):
            np.save(os.path.join(z_dir, "detla_z_{}.npy".format(i)), z)
        print("Finish generating")


    smile_avg_dist = []
    smile_gen_dist = []

    # img = cv2.imread("octocatgurumi.png")
    # img = img[:, :, ::-1]
    # img = cv2.resize(img, (64, 64))
    # img = (torch.tensor(img).permute(2, 0, 1))/256.0
    z_base_male_smiling_1 = graph.generate_z(dataset[0]["x"])
    z_base_male_smiling_2 = graph.generate_z(dataset[1]["x"])
    # imgs_male_smiling.append(img)


    z_base_male_smiling = (z_base_male_smiling_1) + z_base_male_smiling_2
    smile_avg_dist.append(z_base_male_smiling)
    images_male_smiling = []
    names = []
    images_male_smiling.append(run_z(graph, z_base_male_smiling))
    names.append("octogengurumi")
    
    save_images(images_male_smiling, names)
    """