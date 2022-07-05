import torch
import torchvision


def train_fn(generator,
             discriminator,
             loader,
             noise_dimension,
             criterion,
             opt_discriminator,
             opt_generator,
             epochs,
             writer_fake,
             writer_real,
             fixed_noise,
             device='cpu'):

    step = 0
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for batch_idx, (real_image, _) in enumerate(loader):
            batch_size = real_image.shape[0]

            ## Fetching real image and generating FAKE image
            # Real image represents the actual MNIST image
            real_image = real_image.to(device)
            real_image =  torch.argmax(real_image, dim=1) # TODO: Check if correct
            # Noise represents a random matrix of noise used by the generator as input
            noise = torch.randn(batch_size, noise_dimension, 1, 1).to(device)
            fake_image = generator(noise)

            ## Train discriminator, maximize (log(D(real)) + log(1-D(G(z))))
            # log(D(real)) part
            discriminator_real = discriminator(real_image).view(-1)
            loss_discriminator_real = criterion(discriminator_real, torch.ones_like(
                discriminator_real))  # Discriminator should associate real image with  1
            # log(1-D(G(z)) part
            discriminator_fake = discriminator(fake_image).view(-1)
            loss_discriminator_fake = criterion(discriminator_fake, torch.zeros_like(
                discriminator_fake))  # Discriminator should associate fake image with  0
            # Combined loss
            loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2

            discriminator.zero_grad()
            # loss.backward() sets the grad attribute of all tensors with requires_grad=True in the computational graph
            loss_discriminator.backward(
                retain_graph=True)  # retrain_graph so that what was used in this pass is not cleared from cache, ex: fake_image
            opt_discriminator.step()

            ## Train generator, minimize log(1-D(G(z))) OR maximize log(D(G(z)))
            # log(D(G(z)) part
            discriminator_fake = discriminator(fake_image).view(-1)
            loss_generator = criterion(discriminator_fake, torch.ones_like(
                discriminator_fake))  # Generator wants Discriminator to associate fake image with  1

            generator.zero_grad()
            loss_generator.backward(
                retain_graph=True)  # retrain_graph so that what was used in this pass is not cleared from cache, ex: fake_image
            opt_generator.step()

            # Tensorboard code
            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
                          Loss D: {loss_discriminator:.4f}, loss G: {loss_generator:.4f}"
                )

                with torch.no_grad():
                    fake = generator(fixed_noise)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    img_grid_real = torchvision.utils.make_grid(real_image[:32], normalize=True)

                    writer_fake.add_image(
                        "Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real.add_image(
                        "Real Images", img_grid_real, global_step=step
                    )
                    step += 1
            else:
                print(f"Batch {batch_idx}/{len(loader)}")
