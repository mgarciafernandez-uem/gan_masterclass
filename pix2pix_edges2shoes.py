import torch
from torch.utils.data import DataLoader
import os
import tarfile
import urllib.request

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Download and extract edges2shoes dataset
def download_edges2shoes_dataset():
    """Download and extract the edges2shoes dataset"""
    dataset_url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz"
    dataset_dir = "edges2shoes"
    tar_filename = "edges2shoes.tar.gz"
    
    # Create directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download the dataset if not already present
    if not os.path.exists(tar_filename):
        print("Downloading edges2shoes dataset...")
        urllib.request.urlretrieve(dataset_url, tar_filename)
        print("Download completed!")
    
    # Extract the dataset if not already extracted
    if not os.path.exists(os.path.join(dataset_dir, "train")):
        print("Extracting dataset...")
        with tarfile.open(tar_filename, 'r:gz') as tar:
            tar.extractall()
        print("Extraction completed!")
    
    return dataset_dir

# Custom dataset for edges2shoes
class Edges2ShoesDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Assuming images are in train/val folders with paired images
        folder = "train" if is_train else "val"
        self.image_dir = f"{root_dir}/{folder}"
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Load the paired image (edge + shoe side by side)
        image = plt.imread(img_path)
        
        # Split the image into input (edges) and target (shoes)
        h, w, _ = image.shape
        input_img = image[:, :w//2, :]  # Left half (edges)
        target_img = image[:, w//2:, :] # Right half (shoes)
        
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        
        return input_img, target_img

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5) if use_dropout else None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.down1 = nn.Conv2d(3, 64, 4, 2, 1)  # 128x128 -> 64x64
        self.down2 = UNetBlock(64, 128, down=True)  # 32x32
        self.down3 = UNetBlock(128, 256, down=True)  # 16x16
        self.down4 = UNetBlock(256, 512, down=True)  # 8x8
        self.down5 = UNetBlock(512, 512, down=True)  # 4x4
        self.down6 = UNetBlock(512, 512, down=True)  # 2x2
        self.down7 = UNetBlock(512, 512, down=True)  # 1x1
        
        # Decoder
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up4 = UNetBlock(1024, 256, down=False)
        self.up5 = UNetBlock(512, 128, down=False)
        self.up6 = UNetBlock(256, 64, down=False)
        self.final = nn.ConvTranspose2d(128, 3, 4, 2, 1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        # Decoder with skip connections
        u1 = self.up1(d7)
        u2 = self.up2(torch.cat([u1, d6], 1))
        u3 = self.up3(torch.cat([u2, d5], 1))
        u4 = self.up4(torch.cat([u3, d4], 1))
        u5 = self.up5(torch.cat([u4, d3], 1))
        u6 = self.up6(torch.cat([u5, d2], 1))
        
        return torch.tanh(self.final(torch.cat([u6, d1], 1)))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),  # Input + Output channels
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, input_img, target_img):
        return self.model(torch.cat([input_img, target_img], 1))

def train_pix2pix():
    # Download the dataset first
    dataset_path = download_edges2shoes_dataset()
    print(f"Dataset ready at: {dataset_path}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Dataset and dataloader
    train_dataset = Edges2ShoesDataset(dataset_path, transform=transform, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    print(f"Dataset loaded with {len(train_dataset)} training samples")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Training parameters
    num_epochs = 100
    lambda_L1 = 100
    
    print("Starting training...")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters())}")


    loss_g = []
    loss_d = []
    
    # Training loop
    for epoch in range(num_epochs):
        for i, (input_imgs, target_imgs) in enumerate(train_dataloader):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Train Generator
            optimizer_G.zero_grad()
            fake_imgs = generator(input_imgs)
            
            # GAN loss
            pred_fake = discriminator(input_imgs, fake_imgs)
            real_labels = torch.ones_like(pred_fake)
            loss_GAN = criterion_GAN(pred_fake, real_labels)
            
            # L1 loss
            loss_L1 = criterion_L1(fake_imgs, target_imgs)
            
            # Total generator loss
            loss_G = loss_GAN + lambda_L1 * loss_L1
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real images
            pred_real = discriminator(input_imgs, target_imgs)
            real_labels = torch.ones_like(pred_real)
            loss_real = criterion_GAN(pred_real, real_labels)
            
            # Fake images
            pred_fake = discriminator(input_imgs, fake_imgs.detach())
            fake_labels = torch.zeros_like(pred_fake)
            loss_fake = criterion_GAN(pred_fake, fake_labels)
            
            # Total discriminator loss
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(train_dataloader)}] '
                      f'Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}')
                
        # Save sample images every 10 epochs
        if epoch % 10 == 0:
            generator.eval()
            with torch.no_grad():
                # Generate a sample image
                sample_fake = generator(input_imgs[:1])  # Take first image from batch
                
                # Denormalize images for visualization
                def denormalize(tensor):
                    return (tensor * 0.5 + 0.5).clamp(0, 1)
                
                input_denorm = denormalize(input_imgs[0].cpu())
                target_denorm = denormalize(target_imgs[0].cpu())
                fake_denorm = denormalize(sample_fake[0].cpu())
                
                # Create comparison plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(input_denorm.permute(1, 2, 0))
                axes[0].set_title('Input (Edges)')
                axes[0].axis('off')
                
                axes[1].imshow(target_denorm.permute(1, 2, 0))
                axes[1].set_title('Target (Real Shoe)')
                axes[1].axis('off')
                
                axes[2].imshow(fake_denorm.permute(1, 2, 0))
                axes[2].set_title('Generated (Fake Shoe)')
                axes[2].axis('off')
                
                plt.suptitle(f'Epoch {epoch} - Sample Results')
                plt.tight_layout()
                
                # Save the image
                os.makedirs('results', exist_ok=True)
                plt.savefig(f'results/epoch_{epoch}_sample.png', dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Sample image saved for epoch {epoch}")
            generator.train()


        loss_g.append(loss_G.item())
        loss_d.append(loss_D.item())

    # Plot losses
    plt.figure(figsize=(10,5))
    plt.plot(loss_g, label='Generator Loss')
    plt.plot(loss_d, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Pix2Pix Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('pix2pix_loss_plot.png')

        

if __name__ == "__main__":
    train_pix2pix()
