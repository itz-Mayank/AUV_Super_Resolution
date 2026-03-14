"""
 > Training pipeline for FUnIE-GAN (paired) model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer:  https://github.com/itz-mayank
"""

# LIBRARIES
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

# Local imports
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.funiegannew import GeneratorFunieGAN, DiscriminatorFunieGAN
from utils.data_utils import GetTrainingPairs, GetValImage


# CONFIG & ARGUMENTS
parser = argparse.ArgumentParser()

# Path to YAML config
parser.add_argument(
    "--cfg_file",
    type=str,
    default="C:\\Users\\Mayank Meghwal\\Desktop\\Reserch Papers AUV\\AUV\\Code\\FUnIEGAN\\configs\\train_ufo.yaml"
)

parser.add_argument("--epoch", type=int, default=0, help="start epoch")
parser.add_argument("--num_epochs", type=int, default=20, help="total epochs")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.99)

args = parser.parse_args()


# LOAD CONFIG FILE
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

dataset_name = cfg["dataset_name"]
dataset_path = cfg["dataset_path"]   # Already fixed in YAML earlier
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"]
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]


# DIRECTORY PATHS
BASE = "C:\\Users\\Mayank Meghwal\\Desktop\\Reserch Papers AUV\\AUV\\Code\\FUnIEGAN\\"

samples_dir = os.path.join(BASE, "samples", "FunieGAN", dataset_name)
checkpoint_dir = os.path.join(BASE, "checkpoints", "FunieGAN", dataset_name)
model_output_dir = os.path.join(BASE, "models")

os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(model_output_dir, exist_ok=True)


# MODEL, LOSSES, CUDA
Adv_cGAN = torch.nn.MSELoss()
L1_G  = torch.nn.L1Loss()
L_vgg = VGG19_PercepLoss()

lambda_1, lambda_con = 7, 3
patch = (1, img_height // 16, img_width // 16)

generator = GeneratorFunieGAN()
discriminator = DiscriminatorFunieGAN()

is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

if is_cuda:
    generator.cuda()
    discriminator.cuda()
    Adv_cGAN.cuda()
    L1_G.cuda()
    L_vgg.cuda()


# LOAD EXISTING MODEL OR INIT NEW
if args.epoch == 0:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
else:
    gen_path = os.path.join(checkpoint_dir, f"generator_{args.epoch}.pth")
    disc_path = os.path.join(checkpoint_dir, f"discriminator_{args.epoch}.pth")

    generator.load_state_dict(torch.load(gen_path))
    discriminator.load_state_dict(torch.load(disc_path))

    print(f"Loaded checkpoint from epoch {args.epoch}")


optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))


# DATA LOADING
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size=args.batch_size,
    shuffle=True,
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir="validation"),
    batch_size=4,
    shuffle=True,
)


# TRAINING LOOP
for epoch in range(args.epoch, args.num_epochs):
    for i, batch in enumerate(dataloader):

        imgs_distorted = Variable(batch["A"].type(Tensor))
        imgs_good_gt = Variable(batch["B"].type(Tensor))

        valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

        # TRAIN DISCRIMINATOR
        optimizer_D.zero_grad()
        imgs_fake = generator(imgs_distorted)

        loss_real = Adv_cGAN(discriminator(imgs_good_gt, imgs_distorted), valid)
        loss_fake = Adv_cGAN(discriminator(imgs_fake, imgs_distorted), fake)

        loss_D = 0.5 * (loss_real + loss_fake) * 10.0
        loss_D.backward()
        optimizer_D.step()

        # TRAIN GENERATOR
        optimizer_G.zero_grad()
        imgs_fake = generator(imgs_distorted)

        loss_GAN = Adv_cGAN(discriminator(imgs_fake, imgs_distorted), valid)
        loss_1 = L1_G(imgs_fake, imgs_good_gt)
        loss_con = L_vgg(imgs_fake, imgs_good_gt)

        loss_G = loss_GAN + lambda_1 * loss_1 + lambda_con * loss_con
        loss_G.backward()
        optimizer_G.step()

        # Logging
        if not i % 50:
            print(
                f"[Epoch {epoch}/{args.num_epochs}] [Batch {i}/{len(dataloader)}] "
                f"[D loss: {loss_D.item():.3f}] [G loss: {loss_G.item():.3f}]"
            )

        # Validation
        batches_done = epoch * len(dataloader) + i
        if batches_done % val_interval == 0:
            imgs = next(iter(val_dataloader))
            imgs_val = Variable(imgs["val"].type(Tensor))
            imgs_gen = generator(imgs_val)

            img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
            save_image(img_sample, os.path.join(samples_dir, f"{batches_done}.png"), nrow=5, normalize=True)

    # Save model checkpoint
    if epoch % ckpt_interval == 0:
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f"generator_{epoch}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f"discriminator_{epoch}.pth"))


# SAVE FINAL MODELS
torch.save(generator.state_dict(), os.path.join(model_output_dir, "funiegan_generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(model_output_dir, "funiegan_discriminator.pth"))

print("Training completed. Models saved.")
