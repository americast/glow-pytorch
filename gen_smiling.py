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

def save_images(images, names):
    if not os.path.exists("pictures/infer/"):
        os.makedirs("pictures/infer/")
    for img, name in zip(images, names):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite("pictures/smile_1000/{}.png".format(name), img)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # print("Saved as pictures/smile/{}.png".format(name))

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
    for gen in tqdm(range(1000)):
        # Smiling Man


        base_indices_smiling_male = get_base_indices("['Smiling', 'Male']~[]")


        indices_here_smiling_male = get_n_indices(base_indices_smiling_male, 3)



        z_base_male_smiling = []
        imgs_male_smiling = []
        for index in indices_here_smiling_male:
            z_base_male_smiling.append(graph.generate_z(dataset[index]["x"]))
            img = dataset[index]["x"].permute(1, 2, 0).detach().cpu().numpy()
            img = img[:, :, ::-1]
            img = cv2.resize(img, (64, 64))
            imgs_male_smiling.append(img)


        z_base_male_smiling = sum(z_base_male_smiling)/3

        smile_avg_dist.append(z_base_male_smiling)
        # images_male_smiling = []
        # names = []
        # images_male_smiling.append(run_z(graph, z_base_male_smiling))
        # os.system("mkdir -p pictures/smile/"+str(gen))
        # names.append(str(gen)+"/male_smiling_avg")
        
        # save_images(images_male_smiling, names)
        

        # Smiling Woman

        base_indices_smiling_female = get_base_indices("['Smiling']~['Male']")


        indices_here_smiling_female = get_n_indices(base_indices_smiling_female, 3)



        z_base_female_smiling = []
        imgs_female_smiling = []
        for index in indices_here_smiling_female:
            z_base_female_smiling.append(graph.generate_z(dataset[index]["x"]))
            img = dataset[index]["x"].permute(1, 2, 0).detach().cpu().numpy()
            img = img[:, :, ::-1]
            img = cv2.resize(img, (64, 64))
            imgs_female_smiling.append(img)


        z_base_female_smiling = sum(z_base_female_smiling)/3

        # images_female_smiling = []
        # names = []
        # images_female_smiling.append(run_z(graph, z_base_female_smiling))
        # os.system("mkdir -p pictures/smile/"+str(gen))
        # names.append(str(gen)+"/female_smiling_avg")
        
        # save_images(images_female_smiling, names)

        # Neutral Man

        base_indices_neutral_male = get_base_indices("['Male']~['Smiling']")


        indices_here_neutral_male = get_n_indices(base_indices_neutral_male, 3)



        z_base_male_neutral = []
        imgs_male_neutral = []
        for index in indices_here_neutral_male:
            z_base_male_neutral.append(graph.generate_z(dataset[index]["x"]))
            img = dataset[index]["x"].permute(1, 2, 0).detach().cpu().numpy()
            img = img[:, :, ::-1]
            img = cv2.resize(img, (64, 64))
            imgs_male_neutral.append(img)


        z_base_male_neutral = sum(z_base_male_neutral)/3

        # images_male_neutral = []
        # names = []
        # images_male_neutral.append(run_z(graph, z_base_male_neutral))
        # os.system("mkdir -p pictures/smile/"+str(gen))
        # names.append(str(gen)+"/male_neutral_avg")
        
        # save_images(images_male_neutral, names)

        # Neutral Woman

        base_indices_neutral_female = get_base_indices("[]~['Smiling', 'Male']")


        indices_here_neutral_female = get_n_indices(base_indices_neutral_female, 3)



        z_base_female_neutral = []
        imgs_female_neutral = []
        for index in indices_here_neutral_female:
            z_base_female_neutral.append(graph.generate_z(dataset[index]["x"]))
            img = dataset[index]["x"].permute(1, 2, 0).detach().cpu().numpy()
            img = img[:, :, ::-1]
            img = cv2.resize(img, (64, 64))
            imgs_female_neutral.append(img)


        z_base_female_neutral = sum(z_base_female_neutral)/3

        # images_female_neutral = []
        # names = []
        # images_female_neutral.append(run_z(graph, z_base_female_neutral))
        # os.system("mkdir -p pictures/smile/"+str(gen))
        # names.append(str(gen)+"/female_neutral_avg")
        
        # save_images(images_female_neutral, names)

        # Generate smiling man
        z_base_male_smiling_gen = z_base_female_smiling - z_base_female_neutral + z_base_male_neutral

        images_male_smiling_gen = []
        names = []
        images_male_smiling_gen.append(run_z(graph, z_base_male_smiling_gen))
        os.system("mkdir -p pictures/smile_1000")
        names.append("male_smiling_"+str(gen))
        
        save_images(images_male_smiling_gen, names)

        smile_gen_dist.append(z_base_male_smiling_gen)

    f = open(z_dir+"/smile_avg_dist", "wb")
    pickle.dump(smile_avg_dist, f)
    f.close()

    f = open(z_dir+"/smile_gen_dist", "wb")
    pickle.dump(smile_gen_dist, f)
    f.close()

    a = np.array(smile_avg_dist)
    b = np.array(smile_gen_dist)
    a = torch.tensor(a)
    b = torch.tensor(b)
    out = F.kl_div(a, b)
    print("KL divergence: "+str(out))