import torch
import torchvision
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 100
hidden_dim = 256
image_dim = 28 * 28
batch_size = 64
learning_rate = 0.0002
epochs = 50

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, image_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, image_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(latent_dim, hidden_dim, image_dim).to(device)
discriminator = Discriminator(image_dim, hidden_dim).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Lists to store losses
g_losses = []
d_losses = []
import torch
import torchvision
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 100
hidden_dim = 256
image_dim = 28 * 28
batch_size = 64
learning_rate = 0.0002
epochs = 50

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, image_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, image_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(latent_dim, hidden_dim, image_dim).to(device)
discriminator = Discriminator(image_dim, hidden_dim).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Lists to store losses
g_losses = []
d_losses = []

# Training loop
for epoch in range(epochs):
    epoch_g_loss = 0
    epoch_d_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.view(-1, image_dim).to(device)
        batch_size = real_images.size(0)
        
        # Train Discriminator
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Real images
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        
        # Fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
    
    # Calculate average losses for the epoch
    avg_g_loss = epoch_g_loss / num_batches
    avg_d_loss = epoch_d_loss / num_batches
    
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)
    
    print(f'Epoch [{epoch+1}/{epochs}], d_loss: {avg_d_loss:.4f}, g_loss: {avg_g_loss:.4f}')
    
    # Generate sample images
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim).to(device)
            fake_images = generator(z).view(-1, 1, 28, 28)
            
            fig, axes = plt.subplots(4, 4, figsize=(8, 8))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(fake_images[i].squeeze().cpu(), cmap='gray')
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'generated_images_epoch_{epoch+1}.png')
            plt.close()

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), g_losses, label='Generator Loss')
plt.plot(range(1, epochs + 1), d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')

print("Training completed!")
# Training loop
