# Latent-Diffusion-Model-in-Tensorflow-for-Making-Cartoon-Faces
A Latent Diffusion Model demonstration (unconditional) in Tensorflow for making cartoon faces, i.e. anime faces taken from [Kaggle Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset).

A select of 2k images were chosen from the dataset, which are then resized to 128x128 for the target image size. A Variational Autoencoder (VAE) was chosen, whose latent space representation (encoded image) is to be learnt by the diffusion model. This VAE is based on the U-Net architecture without the skips, with the latent dimension to be of 32x32.

<img width="515" height="188" alt="image" src="https://github.com/user-attachments/assets/6d30f018-105f-433f-9f5f-c61f18814859" />

Both the U-Net and the Transformer architecture models have been trained for approximately 1200 epochs.

# Diffusion Model with U-Net Architecture

<img width="1649" height="568" alt="image" src="https://github.com/user-attachments/assets/8696292e-7e70-4219-a6ec-7a70d7023f22" />


# Diffusion Model with Transformer Architecture



Resources that helped me build this project:
[DDPM Paper](https://arxiv.org/abs/2006.11239)
[Diffusion Transformer Paper](https://arxiv.org/abs/2212.09748)
[Kaggle Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
[LPIPS Tensorflow module](https://github.com/moono/lpips-tf2.x)
[DDPM Explanation](https://www.youtube.com/watch?v=H45lF4sUgiE)
[DiT Explanation](https://www.youtube.com/watch?v=aSLDXdc2hkk)
