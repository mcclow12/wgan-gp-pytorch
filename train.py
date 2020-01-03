import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.nn import BCELoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.autograd import grad
from torchvision.utils import save_image
from models import *
from utils import *

from torch.autograd import Variable
class Trainer():
    def __init__(
            self, gen, disc, batch_size, latent_dim, n_disc, disc_lr, 
            gen_lr, beta1, beta2, lam, save_freq, print_freq
            
        ):
        self.data_gen = random_batch_gen(batch_size)
        sample_image = next(self.data_gen)[0][0][0]
        self.image_shape = sample_image.numpy().shape
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.gen = gen 
        self.disc = disc 
        self.n_disc = n_disc
        self.gen_opt = Adam(
                self.gen.parameters(), lr=gen_lr, betas=(beta1, beta2)
        )
        self.disc_opt = Adam(
                self.disc.parameters(), lr=disc_lr, betas=(beta1, beta2)
        )
        self.lam = lam
        self.ep = 0
        self.save_freq = save_freq
        self.print_freq = print_freq
        self.gen_losses = []
        self.disc_losses = []
        self.grad_pens = []

    def train(self, num_epochs):
        while self.ep < num_epochs:
            self._train_epoch()
        self._plot()


    def _train_epoch(self):
        curr_ep = self.ep
        step_num = 0
        while(self.ep == curr_ep):
            self._train_step(step_num)
            step_num += 1

        if (self.ep - 1) % self.save_freq == 0:
            self._save_images()

    def _train_step(self, step_num):
        disc_loss, grad_pen = self._disc_step()
        gen_loss = self._gen_step()
        self._record(step_num, disc_loss, gen_loss, grad_pen)

    def _disc_step(self): 
        self.gen.train()
        self.disc.train()

        freeze_params(self.gen)
        unfreeze_params(self.disc)

        for __ in range(self.n_disc):
            random_batch, self.ep = next(self.data_gen)

            real_images, __ = random_batch
            real_images = preprocess_batch(real_images, flatten=False)
            disc_outputs = self.disc(real_images)

            latents = latent_samples(
                    real_images.shape[0], self.latent_dim)
            generated_images = self.gen(latents)
            gan_outputs = self.disc(generated_images)

            grad_pen = self._grad_pen(real_images, generated_images)

            disc_loss = gan_outputs - disc_outputs + grad_pen 
            disc_loss = disc_loss.mean()

            self.disc_opt.zero_grad()    
            disc_loss.backward()
            self.disc_opt.step()

            return disc_loss, grad_pen


    def _grad_pen(self, real_images, generated_images): 
        epsilon = np.random.uniform(
                size=(generated_images.shape[0], 1, 1, 1))
        epsilon = torch.from_numpy(epsilon).float()
        xhat = epsilon*real_images + (1-epsilon)*generated_images
        xhat.requires_grad_()

        lipschitz_grad = grad(
                outputs=self.disc(xhat).sum(), 
                inputs=xhat,
                create_graph=True, 
                retain_graph = True)[0]
        lipschitz_grad = lipschitz_grad.view(xhat.shape[0], -1)
        grad_norm = torch.sqrt(torch.sum(lipschitz_grad**2, dim=1) 
                                + 1e-12)
        grad_pen = self.lam*(grad_norm - 1)**2
        return grad_pen

    def _gen_step(self): 
        self.gen.train()
        self.disc.train()
        
        unfreeze_params(self.gen)
        freeze_params(self.disc)

        self.gen_opt.zero_grad()    

        latents = latent_samples(self.batch_size, self.latent_dim)
        gan_outputs = self.disc(self.gen(latents))
        
        gen_loss = -gan_outputs.mean()
        gen_loss.backward()

        self.gen_opt.step()
        return gen_loss

    def _record(self, step_num, disc_loss, gen_loss, grad_pen):
        #record statistics 
        self.gen_losses.append(gen_loss.item())
        self.disc_losses.append(disc_loss.item())
        self.grad_pens.append(grad_pen.mean().item())

        if step_num % self.print_freq == 0:
            print('discriminator loss: ', disc_loss.item())
            print('grad_pen', grad_pen.mean().item())
            print('generator loss: ', gen_loss.item())
            print('------------------------')

    def _save_images(self):
        #save image samples
        self.gen.eval()
        self.disc.eval()
        with torch.no_grad():
            latents = latent_samples(self.batch_size, self.latent_dim)
            eval_images = self.gen(latents)

        print('---------')
        print('SAVING')
        print('---------')
        save_images = deprocess_batch(eval_images[:25], 
                                    self.image_shape,
                                    unflatten=False
        )
        save_image(save_images,
                   './generated_images/epoch_{}.png'.format(self.ep),
                   nrow=5)

    def _plot(self):
        #plot losses
        plt.figure()
        plt.plot(self.gen_losses, label='generator loss')
        plt.plot(self.disc_losses, label='discriminator loss')
        plt.plot(self.grad_pens, label='gradient penalty')
        plt.xlabel('batch number')
        plt.ylabel('mean loss for batch')
        plt.legend()
        plt.savefig('./plots/losses.png')
