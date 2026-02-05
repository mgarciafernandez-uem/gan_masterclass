import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Create directories for saving
os.makedirs('results_cgan_mnist/images', exist_ok=True)
os.makedirs('results_cgan_mnist/losses', exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
num_epochs = 100
z_dim = 100
num_classes = 10
image_size = 28 * 28

# Generate and save sample images
def generate_and_save_images(epoch, digit=None, num_samples=16):
    generator.eval()
    with torch.no_grad():
        if digit is None:
            # Generate one image for each digit
            noise = torch.randn(10, z_dim).to(device)
            labels = torch.arange(0, 10).to(device)
            fake_images = generator(noise, labels)
            fake_images = fake_images.view(-1, 1, 28, 28)
            fake_images = (fake_images + 1) / 2  # Denormalize
            
            plt.figure(figsize=(15, 3))
            for i in range(10):
                plt.subplot(2, 5, i + 1)
                plt.imshow(fake_images[i].cpu().squeeze(), cmap='gray')
                plt.title(f'Digit {i}')
                plt.axis('off')
            plt.suptitle(f'Generated MNIST Digits - Epoch {epoch}')
            plt.savefig(f'results_cgan_mnist/images/epoch_{epoch:03d}_all_digits.png', bbox_inches='tight')
            plt.close()
        else:
            # Generate multiple samples for a specific digit
            noise = torch.randn(num_samples, z_dim).to(device)
            labels = torch.full((num_samples,), digit).to(device)
            fake_images = generator(noise, labels)
            fake_images = fake_images.view(-1, 1, 28, 28)
            fake_images = (fake_images + 1) / 2  # Denormalize
            
            plt.figure(figsize=(8, 8))
            for i in range(num_samples):
                plt.subplot(4, 4, i + 1)
                plt.imshow(fake_images[i].cpu().squeeze(), cmap='gray')
                plt.axis('off')
            plt.suptitle(f'Generated MNIST Digit: {digit} - Epoch {epoch}')
            plt.savefig(f'results_cgan_mnist/images/epoch_{epoch:03d}_digit_{digit}.png', bbox_inches='tight')
            plt.close()

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_onehot = torch.zeros(labels.size(0), num_classes).to(labels.device)
        label_onehot.scatter_(1, labels.unsqueeze(1), 1)
        input_tensor = torch.cat([noise, label_onehot], dim=1)
        return self.model(input_tensor)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(image_size + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, labels):
        label_onehot = torch.zeros(labels.size(0), num_classes).to(labels.device)
        label_onehot.scatter_(1, labels.unsqueeze(1), 1)
        input_tensor = torch.cat([image, label_onehot], dim=1)
        return self.model(input_tensor)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Lists to store losses
d_losses = []
g_losses = []

# Training
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        real_images = real_images.view(-1, image_size).to(device)
        labels = labels.to(device)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Real images
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        real_output = discriminator(real_images, labels)
        d_loss_real = criterion(real_output, real_labels)
        
        # Fake images
        noise = torch.randn(real_images.size(0), z_dim).to(device)
        fake_labels = torch.randint(0, num_classes, (real_images.size(0),)).to(device)
        fake_images = generator(noise, fake_labels)
        fake_output = discriminator(fake_images.detach(), fake_labels)
        fake_targets = torch.zeros(real_images.size(0), 1).to(device)
        d_loss_fake = criterion(fake_output, fake_targets)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        fake_output = discriminator(fake_images, fake_labels)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_G.step()
        
        # Store losses
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')
    
    # Save sample images every 10 epochs
    if (epoch + 1) % 10 == 0:
        generate_and_save_images(epoch + 1)

# Save losses
np.save('results_cgan_mnist/losses/d_losses.npy', d_losses)
np.save('results_cgan_mnist/losses/g_losses.npy', g_losses)
# Plot and save loss curves
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.savefig('results_cgan_mnist/losses/loss_curves.png')
plt.close()

# Generate final samples for all digits
for digit in range(10):
    generate_and_save_images(num_epochs, digit)

print("Training complete! Losses and images saved in 'results' folder.")
