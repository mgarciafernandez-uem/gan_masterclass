import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

def train_autoencoder():
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(latent_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training
    model.train()
    loss_history = []
    
    for epoch in range(300):
        total_loss = 0
        for data, _ in train_loader:
            data = data.view(-1, 784).to(device)
            
            optimizer.zero_grad()
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch+1}/300, Loss: {avg_loss:.4f}')
    
    return model, loss_history

def plot_loss_curve(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_latent_space_tsne(model, data_loader, num_samples=2000):
    device = next(model.parameters()).device
    model.eval()
    
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        samples_collected = 0
        for data, label in data_loader:
            if samples_collected >= num_samples:
                break
                
            data = data.view(-1, 784).to(device)
            encoded = model.encode(data)
            
            latent_vectors.append(encoded.cpu().numpy())
            labels.append(label.numpy())
            
            samples_collected += data.shape[0]
    
    # Concatenate all latent vectors and labels
    latent_vectors = torch.from_numpy(np.concatenate(latent_vectors, axis=0)[:num_samples])
    labels = torch.from_numpy(np.concatenate(labels, axis=0)[:num_samples])
    
    # Apply t-SNE
    print("Applying t-SNE to latent representations...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent_vectors.numpy())
    
    # Plot t-SNE results
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig('latent_space_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_samples(model, num_samples=16):
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        # Sample from latent space
        latent_samples = torch.randn(num_samples, 16).to(device)
        generated_images = model.decoder(latent_samples)
        generated_images = generated_images.view(-1, 28, 28)
    
    # Plot generated samples
    plt.figure(figsize=(8, 8))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i].cpu().numpy(), cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('generated_mnist_samples_autoencoder.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Train the autoencoder
    trained_model, loss_history = train_autoencoder()
    
    # Plot training loss
    plot_loss_curve(loss_history)
    
    # Load test data for t-SNE visualization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Visualize latent space with t-SNE
    visualize_latent_space_tsne(trained_model, test_loader)
    
    # Generate fake MNIST samples
    generate_samples(trained_model, num_samples=16)