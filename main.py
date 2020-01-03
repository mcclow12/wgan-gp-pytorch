import os
from train import Trainer
from models import Generator, Discriminator

print_freq = 10 #print frequency (batches)
save_freq = 1 #save frequency (epochs)
num_epochs = 1

batch_size = 64 
latent_dim = 100
disc_lr = 1e-4
gen_lr = 1e-4
beta1 = 0.5
beta2 = 0.9
n_disc = 10 
lam = 10 

os.makedirs('./generated_images', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

gen = Generator(latent_dim)
disc = Discriminator()
trainer = Trainer(gen, disc, batch_size, latent_dim, n_disc,
            disc_lr, gen_lr, beta1, beta2, lam, save_freq, print_freq)

trainer.train(num_epochs)
