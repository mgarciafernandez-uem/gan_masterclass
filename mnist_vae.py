
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE

# Create output directory
os.makedirs('results_vae', exist_ok=True)

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_SIZE = 16
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3

class MNISTVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dimensions=16):
        super().__init__()
        self.latent_dimensions = latent_dimensions
        
        # Encoding layers
        self.encode_layer1 = nn.Linear(784, 400)
        self.encode_layer2 = nn.Linear(400, 200)
        self.mean_layer = nn.Linear(200, latent_dimensions)
        self.variance_layer = nn.Linear(200, latent_dimensions)
        
        # Decoding layers
        self.decode_layer1 = nn.Linear(latent_dimensions, 200)
        self.decode_layer2 = nn.Linear(200, 400)
        self.decode_layer3 = nn.Linear(400, 784)
        
    def encode_input(self, input_data):
        hidden1 = F.relu(self.encode_layer1(input_data))
        hidden2 = F.relu(self.encode_layer2(hidden1))
        mean_output = self.mean_layer(hidden2)
        log_variance = self.variance_layer(hidden2)
        return mean_output, log_variance
    
    def sample_latent(self, mean, log_var):
        standard_dev = torch.exp(0.5 * log_var)
        noise = torch.randn_like(standard_dev)
        return mean + noise * standard_dev
    
    def decode_latent(self, latent_vector):
        hidden1 = F.relu(self.decode_layer1(latent_vector))
        hidden2 = F.relu(self.decode_layer2(hidden1))
        reconstruction = torch.sigmoid(self.decode_layer3(hidden2))
        return reconstruction
    
    def forward(self, input_data):
        mean, log_var = self.encode_input(input_data)
        latent_sample = self.sample_latent(mean, log_var)
        reconstructed = self.decode_latent(latent_sample)
        return reconstructed, mean, log_var

def compute_vae_loss(reconstructed, original, mean, log_variance):
    reconstruction_error = F.binary_cross_entropy(reconstructed, original, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
    return reconstruction_error + kl_divergence

# Data preparation
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup
model = MNISTVariationalAutoencoder(LATENT_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training
loss_history = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)
        optimizer.zero_grad()
        
        reconstructed, mean, log_var = model(data)
        loss = compute_vae_loss(reconstructed, data, mean, log_var)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader.dataset)
    loss_history.append(avg_loss)
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}')

# Generate samples
model.eval()
with torch.no_grad():
    random_latent = torch.randn(64, LATENT_SIZE).to(device)
    generated_samples = model.decode_latent(random_latent)
    generated_samples = generated_samples.view(-1, 1, 28, 28).cpu()
    
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(64):
        axes[i//8, i%8].imshow(generated_samples[i][0], cmap='gray')
        axes[i//8, i%8].axis('off')
    plt.savefig('results_vae/generated_samples.png')
    plt.close()

# Latent space visualization with t-SNE
latent_vectors = []
labels_list = []
model.eval()
with torch.no_grad():
    for data, labels in test_loader:
        data = data.view(-1, 784).to(device)
        mean, _ = model.encode_input(data)
        latent_vectors.append(mean.cpu().numpy())
        labels_list.append(labels.numpy())

latent_data = np.concatenate(latent_vectors)
all_labels = np.concatenate(labels_list)

tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_data)

plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for digit in range(10):
    mask = all_labels == digit
    plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                c=[colors[digit]], label=f'Digit {digit}', alpha=0.6)
plt.legend()
plt.title('t-SNE Visualization of Latent Space')
plt.savefig('results_vae/latent_tsne.png')
plt.close()

# Loss visualization
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title('VAE Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('results_vae/loss_curve.png')
plt.close()

print("Training completed. Results saved in 'results_vae' folder.")