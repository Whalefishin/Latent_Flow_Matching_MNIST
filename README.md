# Latent Flow Matching for MNIST

This repository provides a minimal example for training a flow matching model in a pretrained VAE's latent space to generate MNIST digits. 

## Approach

Denote the data distribution as $P_1$, we first use a pretrained VAE encoder $E(x)$ to transform it into the latent distribution $P_z = E_\ast P_1$, where $\ast$ is the pushforward operation. We then prescribe a flow path between $z \sim P_z$ and $x_0 \sim P_0 = N(0,I)$, then regress on its velocity field $v_\theta(t,x)$ with the following conditional flow matching (CFM) loss:


$$\mathcal{L}_{CFM} = \mathbb{E}_{t\sim U[0,1], x_0 \sim P_z, x_1 \sim P_1} \| v_\theta(t, \psi(x_0|x_1)) - \psi_t'(x_0 | x_1)\|$$.

$$\mathcal{L}_{CFM} = \mathbb{E}_{t\sim U[0,1], x_0 \sim P_z, x_1 \sim P_1} \lVert v(t, \psi(x_0|x_1)) - \psi_t'(x_0|x_1) \rVert^2_2$$

$$\mathbb{E}_{t\sim U[0,1], x_0 \sim P_z, x_1 \sim P_1} \lVert v_\theta(t, \psi(x_0|x_1)) - \psi_t'(x_0|x_1) \rVert^2_2$$

$$\mathbb{E}_{t\sim U[0,1], x_0 \sim P_z, x_1 \sim P_1}$$

$$\lVert v_\theta(t, \psi(x_0|x_1)) - \psi_t'(x_0 | x_1) \rVert^2_2$$

Here, $\psi_t(x_0|x_1)$ is the flow map associated with the prescribed flow. We pick it to be $\psi_t(x|y) = (1-(1-\sigma_{min})t)x + ty$.


## Results


### VAE Reconstruction

As a sanity check, we can visualize the original and reconstructed digits to ensure the pretrained VAE can embed the images as intended.

| Original  | Reconstructed |
| ------------- | ------------- |
| ![result](figures/orig.png)  | ![result](figures/rec.png) |

### Generated Digits

| Epoch 1  | Epoch 25 | Epoch 50 | Epoch 75 | Epoch 100 |
| ------------- | ------------- | ------------- | ------------- |------------- | 
| ![result](figures/orig.png)  | ![result](figures/rec.png) | ![result](figures/orig.png)  | ![result](figures/rec.png) | ![result](figures/rec.png) |


## Dependencies


## Experiments

Simply run:

```
python main.py
```

## Attribution

The diffusion transformer (DiT) code is adapted from the official repository: https://github.com/facebookresearch/DiT

The flow matching model is adapted from https://github.com/gle-bellier/flow-matching/tree/main