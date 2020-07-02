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

def save_images(images, names):
    if not os.path.exists("pictures/infer/"):
        os.makedirs("pictures/infer/")
    for img, name in zip(images, names):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite("pictures/smile/{}.png".format(name), img)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        print("Saved as pictures/smile/{}.png".format(name))

def run_z(graph, z):
    graph.eval()
    x = graph(z=torch.tensor([z]), eps_std=0.3, reverse=True)
    img = x[0].permute(1, 2, 0).detach().cpu().numpy()
    img = img[:, :, ::-1]
    img = cv2.resize(img, (64, 64))
    return img

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



    base_indices = []
    f = open("pic_list/"+"['Smiling', 'Male']~[]", "r")
    while True:
        line = f.readline()
        if not line: break
        base_indices.append(int(line.strip().split("/")[1].split(".")[0]) - 1)
    f.close()

    indices_here = []

    for i in range(3):
        num = int(random() * len(base_indices))
        while base_indices[num] in indices_here:
            num = int(random() * len(base_indices))
        indices_here.append(base_indices[num])


    z_base = []
    imgs = []
    for index in tqdm(indices_here):
        z_base.append(graph.generate_z(dataset[index]["x"]))
        img = dataset[index]["x"].permute(1, 2, 0).detach().cpu().numpy()
        img = img[:, :, ::-1]
        img = cv2.resize(img, (64, 64))
        imgs.append(img)

    # pu.db

    z_base = sum(z_base)/3

    images = []
    names = []
    images.append(run_z(graph, z_base))
    names.append("average")
    
    save_images(images, names)
    save_images(imgs, ["1", "2", "3"])
    # pu.db