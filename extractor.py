"""Train script.

Usage:
    infer_celeba.py <hparams> <dataset_root>
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
import pudb

def select_index(name, l, r, description=None):
    index = None
    while index is None:
        print("Select {} with index [{}, {}),"
              "or {} for random selection".format(name, l, r, l - 1))
        if description is not None:
            for i, d in enumerate(description):
                print("{}: {}".format(i, d))
        try:
            line = int(input().strip())
            if l - 1 <= line < r:
                index = line
                if index == l - 1:
                    index = random.randint(l, r - 1)
        except Exception:
            pass
    return index


def run_z(graph, z):
    graph.eval()
    x = graph(z=torch.tensor([z]), eps_std=0.3, reverse=True)
    img = x[0].permute(1, 2, 0).detach().cpu().numpy()
    img = img[:, :, ::-1]
    img = cv2.resize(img, (256, 256))
    return img


def save_images(images, names):
    if not os.path.exists("pictures/infer/"):
        os.makedirs("pictures/infer/")
    for img, name in zip(images, names):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite("pictures/infer/{}.png".format(name), img)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        print("Saved as pictures/infer/{}.png".format(name))


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    hparams = JsonConfig(hparams)
    dataset_root = args["<dataset_root>"]
    dataset = vision.Datasets["celeba"]
    # set transform of dataset
    transform = transforms.Compose([
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize(hparams.Data.resize),
        transforms.ToTensor()])

    dataset = dataset(dataset_root, transform=transform)
    # interact with user
    # pu.db
    Needs = ["Male", "Smiling"]
    indices = []
    for need in Needs:
        indices.append(dataset.attrs.index(need))

    f = open("pic_list/"+str(Needs), "w")
    for data in dataset.data:
        choose = True

        for index in indices:
            if data["attr"][index] != 1:
                choose = False

        if choose:
            f.write(data["path"]+"\n")

    f.close()
