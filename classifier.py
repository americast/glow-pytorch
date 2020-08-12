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
from PIL import Image
from torch import optim
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from classifier_utils import *

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


  

if __name__ == "__main__":

    EPOCHS = 100    
    LR = 1e-5

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

    f = open("pic_list/['Smiling', 'Male']~[]", "r")
    male_smiling = []
    while True:
        line = f.readline()
        if not line: break
        male_smiling.append(line.strip())
    f.close()

    hparams = JsonConfig("hparams/celeba.json")
    # set transform of dataset
    transform = transforms.Compose([
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize(hparams.Data.resize),
        transforms.ToTensor()])

    transformed_dataset = classifier_data(male_neutral, female_neutral, male_smiling, female_smiling, transform=transform, cut_len=1000)

    dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=True, num_workers=32)

    model = enc_classifier().cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for E in tqdm(range(EPOCHS)):
        print("\n")
        losses = []
        c = 0
        for data, n, r in dataloader:
            data, n, r = data.to("cuda"), n.to("cuda"), r.to("cuda")
            c+=1
            _, y = model(data, n)
            out = loss(y, r)
            # pu.db
            print(str(c)+"/"+str(len(transformed_dataset))+"; "+"r: "+str(r)+"; loss: "+str(out)+"      ", end="\r")
            optimizer.zero_grad()

            out.backward()

            optimizer.step()

            losses.append(out)

        print()
        loss_here = sum(losses)/len(losses)
        print("Avg loss in epoch "+str(E)+": "+str(loss_here))
        if E == 0:
            avg_loss = loss_here

        if loss_here <= avg_loss:
            avg_loss = loss_here
            torch.save(model.state_dict(), "./classifier_model.pt")
            f = open("classifier_model_details", "w")
            f.write("loss: "+str(loss_here)+"\nEpoch: "+str(E)+"\n")
            print("Model saved!")





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

