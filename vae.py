import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import multivariate_normal
from torch.utils.data import random_split
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch

from torch.utils.data import ConcatDataset
from torch.optim import lr_scheduler
from scipy.stats import norm

torch.manual_seed(0)

# Custom dataset to load only specific digits from an npz file
class SubsetImagesFromNPZ(Dataset):
    def __init__(self, npz_file, keep_labels=[1, 4, 8], noise_std=0, augment=False, augment_ratio=1):
        data = np.load(npz_file)
        self.images = data['data']
        self.labels = data['labels']
        self.keep_labels = keep_labels
        self.noise_std = noise_std
        self.augment = augment
        self.augment_ratio = augment_ratio

        # Filter data based on labels
        self.filtered_data = [(img, lbl) for img, lbl in zip(self.images, self.labels) if lbl in keep_labels]

        # Define transforms for data augmentation if enabled
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomRotation(30)
        ]) if augment else None

        # Prepare the final dataset with originals and augmented images
        self.final_data = []
        for img, lbl in self.filtered_data:
            # Always include the original image
            self.final_data.append((torch.tensor(img, dtype=torch.float32).view(-1) / 255.0, lbl))  # Flattened

            # Add augmented versions
            if augment:
                for _ in range(augment_ratio):
                    aug_img = self.apply_transform(img)
                    self.final_data.append((aug_img, lbl))

    def apply_transform(self, img):
        # Reshape img to (1, H, W) for applying 2D transforms
        img_tensor = torch.tensor(img, dtype=torch.float32).view(1, int(np.sqrt(img.size)), int(np.sqrt(img.size))) / 255.0
        aug_img = self.transform(img_tensor) if self.transform else img_tensor
        return aug_img.view(-1)  # Flatten to (H*W,) or (1, H*W)

    def __len__(self):
        return len(self.final_data)

    def __getitem__(self, idx):
        img, lbl = self.final_data[idx]

        # Apply noise if specified
        noisy_img = img + torch.randn_like(img) * self.noise_std
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)

        # Ensure the image is flattened (H*W,) or (1, H*W) if needed
        return noisy_img.view(-1), lbl


# VAE model
class VAE(nn.Module):
    def __init__(self, img_shape=(1, 28, 28), latent_dim=2, dropout_rate=0.2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Dropout after convolutional layers
        self.dropout_conv = nn.Dropout2d(self.dropout_rate)
        
        # Flatten layer dimensions after conv
        self.flatten_shape = (256, 7, 7) 
        self.fc1 = nn.Linear(np.prod(self.flatten_shape), 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_log_sigma = nn.Linear(512, latent_dim)
        
        # Dropout after fully connected layer
        self.dropout_fc = nn.Dropout(self.dropout_rate)
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, np.prod(self.flatten_shape))
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)
        
        # Dropout after deconvolutional layers
        self.dropout_deconv = nn.Dropout2d(self.dropout_rate)
    
    def encode(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)

        x = F.relu(self.conv1(x))
        x = self.dropout_conv(F.relu(self.conv2(x)))
        x = self.dropout_conv(F.relu(self.conv3(x)))
        x = self.dropout_conv(F.relu(self.conv4(x)))
        x = self.dropout_conv(F.relu(self.conv5(x)))
        x = x.view(-1, np.prod(self.flatten_shape))
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        mu = self.fc_mu(x)
        log_sigma = self.fc_log_sigma(x)
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc2(z))
        x = x.view(-1, *self.flatten_shape)
        x = self.dropout_deconv(F.relu(self.deconv1(x)))
        x = self.dropout_deconv(F.relu(self.deconv2(x)))
        x = self.dropout_deconv(F.relu(self.deconv3(x)))
        x = self.dropout_deconv(F.relu(self.deconv4(x)))
        x = torch.sigmoid(self.deconv5(x))
        return x

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        x_reconstructed = self.decode(z)
        
        x_reconstructed = x_reconstructed.view(-1, 1, 28, 28)
        return x_reconstructed, mu, log_sigma

class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_ = None
        self.covs_ = None
        self.weights_ = None

    def initialize_parameters(self, latent_vectors, labels):
        unique_labels = np.unique(labels)
        means = np.array([latent_vectors[labels == label].mean(axis=0) for label in unique_labels])
        n_clusters = len(unique_labels)
        latent_dim = means.shape[1]
        covs = np.array([np.eye(latent_dim) for _ in range(n_clusters)])
        weights = np.ones(n_clusters) / n_clusters
        return means, covs, weights


    def gaussian_pdf(self, X, mean, cov):
        try:
            return multivariate_normal(mean=mean, cov=cov).pdf(X)
        except np.linalg.LinAlgError:
            regularized_cov = cov + 1e-6 * np.eye(cov.shape[0])
            return multivariate_normal(mean=mean, cov=regularized_cov).pdf(X)

    def e_step(self, X,means,covs,weights):
        n_samples = X.shape[0]
        pdfs = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            pdfs[:, k] = weights[k] * self.gaussian_pdf(X, means[k], covs[k])
        responsibilities = pdfs / (np.sum(pdfs, axis=1, keepdims=True) + 1e-9)
        return responsibilities
    
    def m_step(self, latent_vectors, prob):
        N_k = prob.sum(axis=0)
        weights = N_k / latent_vectors.shape[0]
        means = (prob.T @ latent_vectors) / N_k[:, np.newaxis]
        n_clusters = len(N_k)
        latent_dim = latent_vectors.shape[1]
        covs = np.zeros((n_clusters, latent_dim, latent_dim))

        for k in range(n_clusters):
            diff = latent_vectors - means[k]
            weighted_diff = prob[:, k][:, np.newaxis] * diff
            covs[k] = (weighted_diff.T @ diff) / N_k[k] + 1e-6 * np.eye(latent_dim)

        return means, covs, weights

    def log_likelihood(self, X, means, covs, weights):
        pdfs = self.gaussian_pdf(X, means, covs)
        weighted_pdfs = pdfs * weights
        log_likelihood = np.sum(np.log(weighted_pdfs.sum(axis=1) + 1e-9))
        return log_likelihood / X.shape[0]

    def fit(self, latent_vectors, means, covs, weights):
        for iteration in range(self.max_iter):
            prob = self.e_step(latent_vectors, means, covs, weights)
            new_means, new_covs, new_weights = self.m_step(latent_vectors, prob)
            if np.linalg.norm(new_means - means) < self.tol:
                print(f"GMM converged after {iteration + 1} iterations")
                break
            means, covs, weights = new_means, new_covs, new_weights

        print("GMM converged")
        return means, covs, weights

    def predict(self, X, means, covs, weights):
        pdfs = self.gaussian_pdf(X, means, covs)
        weighted_pdfs = pdfs * weights
        return np.argmax(weighted_pdfs, axis=1)

# Loss function combining reconstruction and KL divergence
def loss_function(recon_x, x, mu, logvar, beta=0.5):
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta*KLD

# Training VAE
def train_vae(train_loader, val_loader, save_path, gmm_save_path):
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Constant initial learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
    
    num_epochs = 200
    epochs_without_improvement = 0
    patience = 15
    best_val_loss = float('inf')

    train_size = int(0.8 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    train_subset, val_subset = random_split(train_loader.dataset, [train_size, val_size])

    combined_val_dataset = ConcatDataset([val_subset, val_loader.dataset])

    # Create loaders
    train_subset_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_subset_loader = DataLoader(combined_val_dataset, batch_size=64, shuffle=False)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_subset_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        # Calculate validation loss
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data, _ in val_subset_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                val_loss += loss.item()
        val_loss /= len(val_subset_loader.dataset)

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)  # Save the best model
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        # Log the losses and current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_subset_loader.dataset)}, Validation Loss: {val_loss}, Learning Rate: {current_lr}")

    train_latent_vectors, train_labels = extract_latent_vectors(model, train_loader)
    train_gmm(train_latent_vectors, train_labels,val_loader, model,gmm_save_path)
    # Train the GMM using the initialized centers from the validation set but fitting on the training set

def plot_2d_manifold(vae, latent_dim=2, n=20, digit_size=28, device='cuda'):
    figure = np.zeros((digit_size * n, digit_size * n))

    # Generate a grid of values between 0.05 and 0.95 percentiles of a normal distribution
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    vae.eval()  # Set VAE to evaluation mode
    with torch.no_grad():
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], device=device).float()
                decoded_img = vae.decode(z_sample)
                digit = decoded_img[0, 0].cpu().numpy()

                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit


    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gnuplot2')
    plt.axis('off')
    plt.show()
    plt.savefig('plot_2d')

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_latent_vectors(model, data_loader, is_validation=False):
    model.eval()
    latent_vectors = []
    labels = []
    class_latents = {}

    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            _, mu, _ = model(data)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(label.numpy())

            if is_validation:
                for i in range(len(label)):
                    lbl = label[i].item()
                    if lbl not in class_latents:
                        class_latents[lbl] = []
                    class_latents[lbl].append(mu[i].cpu().numpy())

    if is_validation:
        # Return class-separated latents for cluster initialization
        return class_latents
    else:
        return np.concatenate(latent_vectors), np.concatenate(labels)

from scipy.optimize import linear_sum_assignment
from collections import defaultdict
def train_gmm(train_latent_vectors, train_labels,val_loader, model,gmm_save_path, n_components=3):
    class_latents = extract_latent_vectors(model, val_loader, is_validation=True)
    val_latents,val_labels = extract_latent_vectors(model, val_loader, is_validation=False)

    gmm = GMM(n_components=n_components)
    means,covs,weights=gmm.initialize_parameters(val_latents,val_labels)
    means,covs,weights=gmm.fit(train_latent_vectors,means,covs,weights)
    gmm_preds = []
    for X in val_latents:
        prob = []
        for i in range(len(means)):
            pdf_value = gmm.gaussian_pdf(X, means[i], covs[i])
            prob.append(pdf_value)
        prediction = np.argmax(prob)
        gmm_preds.append(prediction)

    gmm_preds = np.array(gmm_preds)  
    cluster_label_mapping = {}
    for i in range(len(val_labels)):
        true_label = val_labels[i]
        predicted_cluster = gmm_preds[i]
        if predicted_cluster not in cluster_label_mapping:
            cluster_label_mapping[predicted_cluster] = []
        cluster_label_mapping[predicted_cluster].append(true_label)

    cluster_to_class = {}
    for cluster in cluster_label_mapping:
        labels = cluster_label_mapping[cluster]
        cluster_to_class[cluster] = max(set(labels), key=labels.count)

    with open(gmm_save_path, "wb") as f:
        pickle.dump((gmm,cluster_to_class,means,covs,weights), f)
    visualize_gmm(train_latent_vectors,train_labels,means,covs)
    
from matplotlib.patches import Ellipse
# Visualize GMM clusters

def visualize_gmm(latent_vectors, labels,means,covs, output_path='gmm_output.png'):
    color_scheme = ['red', 'blue', 'green']
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels, cmap='plasma', s=15, alpha=0.2)

    for idx, (mean, cov) in enumerate(zip(means, covs)):
        eigvals, eigvecs = np.linalg.eigh(cov)
        axes_lengths = np.sqrt(eigvals) * 2
        orientation = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        cluster_ellipse = Ellipse(xy=mean, width=axes_lengths[0], height=axes_lengths[1], angle=orientation,
                                  edgecolor=color_scheme[idx], facecolor='none', linestyle=':', lw=2)
        ax.add_patch(cluster_ellipse)
        ax.plot(mean[0], mean[1], 'o', color=color_scheme[idx], markersize=10, label=f"Center of Cluster {idx + 1}")

    # Set the title, labels, and grid
    ax.set_title('GMM Cluster Visualization with Ellipses')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    ax.legend(loc="best")
    ax.grid(True, linestyle='-.', alpha=0.8)

    plt.savefig(output_path)
    plt.show()

# Evaluate GMM performance
def evaluate_gmm_performance(labels_true, labels_pred):
    accuracy = accuracy_score(labels_true, labels_pred)
    precision_macro = precision_score(labels_true, labels_pred, average='macro')  # Macro precision
    recall_macro = recall_score(labels_true, labels_pred, average='macro')  # Macro recall
    f1_macro = f1_score(labels_true, labels_pred, average='macro')  # Macro F1

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

def show_reconstruction(model, val_loader, n=15):
    model.eval()
    data, labels = next(iter(val_loader))
    
    data = data.to(device)
    recon_data, _, _ = model(data)
    reconstructed_images = [] 
    
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    avg = 0
    for i in range(n):
        original_img = data[i].cpu().numpy().reshape(28, 28)
        reconstructed_img = recon_data[i].cpu().view(28, 28).detach().numpy()

        # Calculate SSIM score
        ssim_score = ssim(original_img, reconstructed_img, data_range=original_img.max() - original_img.min())
        print(f"Image {i + 1} SSIM score: {ssim_score:.4f}")
        avg += ssim_score

        # Original images
        axes[0, i].imshow(original_img, cmap='gray')
        axes[0, i].axis('off')
        
        # Reconstructed images
        axes[1, i].imshow(reconstructed_img, cmap='gray')
        axes[1, i].axis('off')
        reconstructed_images.append(reconstructed_img)
    avg = avg/15
    print(f"Average : {avg:.4f}")
    # Save the figure instead of showing it
    img_path = "reconstructed_total.png"
    plt.savefig(img_path)
    plt.close(fig)
    print(f"Reconstruction image saved at {img_path}")
    reconstructed_images = np.array(reconstructed_images)
    np.savez_compressed('vae_reconstructed.npz', reconstructed_images=reconstructed_images)
    print(f"Reconstructed images saved to 'vae_reconstructed.npz')")

# Class prediction during testing
def test_classifier(test_loader, model_path, gmm_params_path, save_path='vae.csv'):
    model = load_model(model_path)
    with open(gmm_params_path, 'rb') as f:
        gmm, gmm_to_true_label_map,means,covariances,p1 = pickle.load(f)
    
    test_data = np.load(test_loader)['data']
    test_dataset = SubsetImagesFromNPZ(npz_file=test_loader, noise_std=0, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    latent_vectors, true_labels = extract_latent_vectors(model, test_loader)
    gmm_preds = []
    for vector in latent_vectors:
        # Calculate the likelihood of this vector belonging to each cluster
        likelihoods = [gmm.gaussian_pdf(vector, mean, covariance) for mean, covariance in zip(means, covariances)]
        predicted_label = np.argmax(likelihoods)
        gmm_preds.append(predicted_label)
    # Remap GMM predictions to true labels
    remapped_preds = np.array([gmm_to_true_label_map[pred] for pred in gmm_preds])

    # Evaluate performance
    metrics = evaluate_gmm_performance(true_labels, remapped_preds)
    print("GMM Performance Metrics:", metrics)
    
    # Save predictions
    with open(save_path, 'w') as f:
        f.write("Predicted_Label\n")
        for label in remapped_preds:
            f.write(f"{label}\n")


# Load the trained model
def load_model(model_path):
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_path))
    return model


# Main function to parse commands
if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3] if len(sys.argv) > 3 else None
    arg4 = sys.argv[4] if len(sys.argv) > 4 else None
    arg5 = sys.argv[5] if len(sys.argv) > 5 else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) == 4:  # VAE reconstruction
        path_to_test_dataset_recon = arg1
        test_reconstruction_func = arg2
        vaePath = arg3

        test_dataset = SubsetImagesFromNPZ(npz_file=path_to_test_dataset_recon, noise_std=0, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = load_model(vaePath)
        print(f"Number of trainable parameters in MLPVAE: {count_parameters(model)}")
        plot_2d_manifold(model, latent_dim=2, n=20, digit_size=28, device='cuda')
        show_reconstruction(model, test_loader)

    elif len(sys.argv) == 5:  # Class prediction during testing
        path_to_test_dataset = arg1
        test_classifier_func = arg2
        vaePath = arg3
        gmmParamsPath = arg4

        test_classifier(path_to_test_dataset, vaePath, gmmParamsPath)

    else:
        path_to_train_dataset = arg1
        path_to_val_dataset = arg2
        trainStatus = arg3
        vaePath = arg4
        gmmParamsPath = arg5

        train_dataset = SubsetImagesFromNPZ(npz_file=path_to_train_dataset, noise_std=0, augment=False)
        val_dataset = SubsetImagesFromNPZ(npz_file=path_to_val_dataset, noise_std=0, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        train_vae(train_loader, val_loader, vaePath, gmmParamsPath)

