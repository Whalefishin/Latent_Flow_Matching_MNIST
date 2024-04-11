import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from zuko.utils import odeint
from tqdm import tqdm
from typing import *
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from DiT import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OTFlowMatching:
    def __init__(self, sig_min: float = 0.001) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.eps = 1e-5

    def psi_t(self, x, x_1, t):
        # we have freedom in choosing the flow map psi_t(x,x_1) = a_t * x + b_t * x_1 
        # as long as it satisfies a_0 = 0, a_1 = 1, b_0 = 1, b_1 = 1.
        # Here, we picked the OT flow map.
        
        return (1 - (1 - self.sig_min) * t) * x + t * x_1

    def loss(self, v_t, x_1):
        """ Compute loss
        """
        # t ~ Unif([0, 1])
        t = (torch.rand(1, device=x_1.device) + torch.arange(len(x_1), device=x_1.device) / len(x_1)) \
            % (1 - self.eps)
        t = t[:, None, None, None].expand(x_1.shape)
        # x ~ p_0
        x_0 = torch.randn_like(x_1)
        # first term in CFM
        v_psi = v_t(t[:,0,0,0], self.psi_t(x_0, x_1, t))
        # second term in CFM
        d_psi = x_1 - (1 - self.sig_min) * x_0

        return torch.mean((v_psi - d_psi) ** 2)
  

class CondVF(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t, x):
        return self.net(t, x)
        
    def wrapper(self, t, x):
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x)

    def decode_t0_t1(self, x_0, t0, t1):
        return odeint(self.wrapper, x_0, t0, t1, self.parameters())

    def encode(self, x_1):
        # This method is not used, because we would've used it for training the CNF 
        # by calling this method and backprop over the ODEINT function (neuralODE style).
        # However, we now train with a different objective (conditional flow matching), so we no longer need this.
        return odeint(self.wrapper, x_1, 1., 0., self.parameters())

    def decode(self, x_0):
        # This method is used for generation. We will first sample x_0 ~ P_0, then
        # generate x_1 by integrating v_theta from 0 to 1 with x_0 as the IC.
        return odeint(self.wrapper, x_0, 0., 1.)


class FM(nn.Module):
    def __init__(self, v_t, flow, d, h, w):
        super().__init__()
        self.v_t = v_t
        self.flow = flow
        self.d = d
        self.h = h
        self.w = w

    def get_loss(self, x):
        # x ~ P_1, a real data point.
        return self.flow.loss(self.v_t, x)

    def sample(self, n):
        # sample from the base dist, P_0 = N(0,I)
        x_0 = torch.randn(n, self.d, self.h, self.w).to(device)
        # integrate over the learned velocity field to generate a sample from P_1
        x_1 = v_t.decode(x_0)
        
        return x_1


def get_data(batch_size):
    tf = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5)),
            transforms.Lambda(lambda x: x.expand(1,-1,-1)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)) # convert to RGB
        ]
    )

    X_train = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    X_val = MNIST(
        "./data",
        train=False,
        download=True,
        transform=tf,
    )
    train_dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(X_val, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader


d,h,w = 4,4,4 # VAE latent dimension
batch_size = 32
n_samples = 16
img_size = 32
latent_size = img_size // 8
train_dataloader, val_dataloader = get_data(batch_size)

# VAE used for image embedding
# To see that the pretrained VAE can indeed reconstruct the images, try: 
# x_rec = vae.decode(vae.encode(x_1).latent_dist.sample()).sample
# where x_1 is a batch of image with shape (B,C,H,W)
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

# Flow matching models
model = OTFlowMatching()
net = DiT_S_4(input_size=latent_size, num_classes=10, learn_sigma=False).to(device)
v_t = CondVF(net)
fm  = FM(v_t, model, d, h, w)

losses = [] 
optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-4)
n_epochs = 500
    
# training
for epoch in range(n_epochs):
    for batch in tqdm(train_dataloader):
        # get data
        x_1 = batch[0].to(device) # (B,c,H,W)
        # c   = batch[1].to(device)
        # encode image as latent vectors + normalize
        x_1 = vae.encode(x_1).latent_dist.sample().mul_(0.18215) # (B,d,h,w) = (B,4,4,4)

        # compute loss 
        loss = fm.get_loss(x_1)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += [float(loss)]

    # logging
    print ("Epoch: {}, avg loss: {:.4f}".format(epoch, np.mean(np.array(losses))))
    losses = []
    # Sampling
    with torch.no_grad():
        # get a sample from the VAE's latent dist
        latent_samples = fm.sample(n_samples)
        # decode the sample into an image
        samples = vae.decode(latent_samples / 0.18215).sample
        save_image(samples, "epoch{}.png".format(epoch), nrow=4, normalize=True, value_range=(-1, 1))

# Sampling
with torch.no_grad():
    latent_samples = fm.sample(n_samples)
    samples = vae.decode(latent_samples / 0.18215).sample
    save_image(samples, "samples_final.png", nrow=4, normalize=True, value_range=(-1, 1))
