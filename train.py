import torch
import os
import matplotlib.pyplot as plt
from torch import nn
from tqdm.auto import tqdm
from datasets import MNISTDataModule
from models import Generator, Discriminator
from utils import *

# Setup
z_dim = 64
mnist_shape = (1, 28, 28)
n_classes = 10
lr = 0.0002
batch_size = 128
n_epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
display_step = 500

# Initialize data
data_module = MNISTDataModule(batch_size)
train_loader = data_module.train_dataloader()

# Calculate input dimensions
generator_input_dim, discriminator_image_channel = calculate_input_dim(z_dim, mnist_shape, n_classes)

# Model initialization
generator = Generator(input_dim=generator_input_dim).to(device)
discriminator = Discriminator(image_channel=discriminator_image_channel).to(device)

# Initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# Optimizers and loss
criterion = nn.BCEWithLogitsLoss()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Training loop
cur_step = 0
generator_losses = []
discriminator_losses = []

savedir="generated_images"
os.makedirs(savedir, exist_ok=True)

for epoch in range(n_epochs):
    for real, labels in tqdm(train_loader):
        real = real.to(device)
        labels = labels.to(device)
        batch_size = real.size(0)

        # One-hot encode the labels
        one_hot_labels = one_hot_encoder_vector_from_labels(labels, n_classes)
        image_one_hot_labels = one_hot_labels[:, :, None, None].repeat(1, 1, mnist_shape[1], mnist_shape[2])

        # Create noise vector and fake labels
        noise = torch.randn(batch_size, z_dim, device=device)
        noise_and_labels = concat_vectors(noise, one_hot_labels)
        fake_images = generator(noise_and_labels)

        #########################
        # Train Discriminator
        #########################
        disc_optimizer.zero_grad()

        fake_input = concat_vectors(fake_images.detach(), image_one_hot_labels)
        real_input = concat_vectors(real, image_one_hot_labels)

        disc_fake_pred = discriminator(fake_input)
        disc_real_pred = discriminator(real_input)

        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        disc_loss.backward()
        disc_optimizer.step()

        #########################
        # Train Generator
        #########################
        gen_optimizer.zero_grad()
        fake_input = concat_vectors(fake_images, image_one_hot_labels)
        disc_fake_pred = discriminator(fake_input)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

        gen_loss.backward()
        gen_optimizer.step()

        # Logging and plotting
        generator_losses.append(gen_loss.item())
        discriminator_losses.append(disc_loss.item())

        if cur_step % display_step == 0:
            print(f" Step {cur_step}: Gen loss: {gen_loss:.4f} | Disc loss: {disc_loss:.4f}")

            # Plot real and fake images in 5x5 grid
            #plot_images_grid(real, fake_images)
            filename = os.path.join(savedir, f"epoch_{epoch+1}_step_{cur_step}.png")
            save_images_grid(real, fake_images, filename)
        cur_step += 1

    print(f"Epoch {epoch + 1}/{n_epochs} completed.")

# Final loss plot
loss_plot_dir = "loss_plots"
os.makedirs(loss_plot_dir, exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label="Generator Loss")
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
loss_plot_file = f"{loss_plot_dir}/training_losses.png"
plt.savefig(loss_plot_file)
plt.show()
print(f"Loss plot saved to {loss_plot_file}")