# NOT COMPLETED
options under consideration: Wasserstein GAN, CelebA, Use new dataset, FID
# DCGAN-pytorch
Implementation of Deep Convolutional Generative Adversarial Networks<br>
Based on paper: Unsupervised representation learning with deep convolutional generative adversarial networks<br>


# DCGAN Model Architecture
![image](https://user-images.githubusercontent.com/61140071/101329973-69313280-38b5-11eb-876d-e88e3e8a47ad.png)
https://arxiv.org/pdf/1511.06434.pdf
# Install

# Prerequisites

# Development Environments

# Results
## Generated image by same noise during training
![Alt Text](https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/mnist_z_gen.gif)

## Generated image after 30 epoch training
<img src="https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/gen_image.JPG" width="360" height="360">

## Interpolation
<img src="https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/interpolate.png" width="360" height="360">

## Generator Loss Tracking
<img src="https://user-images.githubusercontent.com/61140071/101358054-a8737980-38dd-11eb-9932-a676d109b2d4.png" width="360" height="360">
(loss is calculated by evaluating mean of Generator and Discriminator)

# Reference
[1] Alec Radford & Luke Metz, Soumith Chintala.(2016). 'Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks' arXiv:1511.06434v2<br>
[2] Ian J. Goodfellow et al. (2014).'Generative Adversarial Networks' arXiv:1406.2661v1<br>
[3] Martin Arjovsky et al. (2017). 'Wasserstein GAN' arXiv:1701.07875<br>
