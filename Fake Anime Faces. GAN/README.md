# Anime Faces generator. Generative adversarial networks

In this [project](https://github.com/privet1mir/Puppy-projects/blob/main/Fake%20Anime%20Faces.%20GAN/fake_anime_faces.ipynb) I will train GAN to generate anime faces.

As a training dataset I'll use [Anime Face Dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) from kaggle, which has 21,551 "high-quality" anime faces. Original dataset is big enough, so we'll use just a part of it.


<img src='https://github.com/privet1mir/Puppy-projects/blob/main/Fake%20Anime%20Faces.%20GAN/images/anime_real.png' width=500>

## GAN Structure.

When we construct descriminator and generator we need to remember that:

* Discriminator's input is an image (tensor size: `3 x image_size x image_size`) and return the probability that the image is real (tensor of rank 1)

* Generator's input is a tensor of noises, it's size: `latent_size x 1 x 1` and generates images with size: `3 x image_size x image_size`

In our case model's noise input looks like this: 

<img src='https://github.com/privet1mir/Puppy-projects/blob/main/Fake%20Anime%20Faces.%20GAN/images/noise.png' width=300>

Remainder of GAN's architecture:

<img src='https://github.com/privet1mir/Puppy-projects/blob/main/Fake%20Anime%20Faces.%20GAN/images/The-structure-of-the-GAN.png' width=700>

If we want to train our own model the algorithm is the following:

1. We train discriminator:
  * Take real images and set it's labels == 1
  * Generate images by generator and set it's labels == 0
  * Train binary classifier

2. We train generator:
  * Generate images by generator and set it's labels == 0
  * Predict if it's real or not by discriminator 


As a loss function we will use binary cross-entropy (aka BCELOSS from torch): 


$\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]$,

where $N$ is the batch size. 

## Structure of Neural Networks used for this task.

Discriminator:

<img src='https://github.com/privet1mir/Puppy-projects/blob/main/Fake%20Anime%20Faces.%20GAN/images/discriminator%20structure.png' width=500>

Generator:

<img src='https://github.com/privet1mir/Puppy-projects/blob/main/Fake%20Anime%20Faces.%20GAN/images/generator%20structure.png' width=500>


## Loss/epoch plot and at the score graph.


<img src='https://github.com/privet1mir/Puppy-projects/blob/main/Fake%20Anime%20Faces.%20GAN/images/losses.png' width=1100>


<img src='https://github.com/privet1mir/Puppy-projects/blob/main/Fake%20Anime%20Faces.%20GAN/images/scores.png' width=1100>

Last graph also needs to say some words about. We can see that "Real score" is close to one, means that our classifier predicts almost all images with true labels (which means true images, not generated ones). The same thing we can say about "Fake" score, which is collected for fake images (means generated from noise). 

## Final result. Generated (from noise) image.


<img src='https://github.com/privet1mir/Puppy-projects/blob/main/Fake%20Anime%20Faces.%20GAN/images/anime_fake.png' width=600>

The result is quite good for such simple model (small data that I used, also not complex network). In later investigations I've some plans to dive deeper in this interesting field of image generating. 
