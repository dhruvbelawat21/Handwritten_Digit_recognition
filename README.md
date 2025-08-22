Handwritten Digit Recognition with VAE + GMM
This project implements a Variational Autoencoder (VAE) combined with a Gaussian Mixture Model (GMM) for handwritten digit recognition, clustering, and reconstruction. Unlike traditional classification pipelines, this approach learns a low-dimensional latent representation of digits and leverages unsupervised clustering for label prediction.

✨ Features
Custom Dataset Loader: Load .npz datasets with support for filtering specific digits, noise injection, and optional data augmentation.

Variational Autoencoder (VAE):
Encoder–Decoder architecture using CNNs.
Latent space sampling with reparameterization trick.
KL-Divergence + Reconstruction loss with adjustable β-VAE factor.

Gaussian Mixture Model (GMM):
Trained on latent vectors from VAE.
Cluster-to-class mapping for digit recognition.
Ellipse visualization of latent clusters.

Visualization:
Reconstructed vs. original digits with SSIM scores.
2D latent manifold generation.
Cluster visualization with covariance ellipses.
Evaluation: Accuracy, Precision, Recall, F1-score (macro).

Training Utilities:
Learning rate scheduling & early stopping.
Parameter count inspection.
Saving/loading trained models and GMM parameters.

📂 Repository Structure
SubsetImagesFromNPZ: Custom dataset loader.
VAE: Variational Autoencoder model.
GMM: Gaussian Mixture Model implementation.
train_vae(): Train VAE + GMM pipeline.

test_classifier(): Evaluate recognition performance.

visualize_gmm(), plot_2d_manifold(): Visualization utilities.

show_reconstruction(): Generate and save reconstructions.
