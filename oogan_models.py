import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.distributions.multivariate_normal import MultivariateNormal as multi_normal
from torchvision import utils as vutils
from itertools import chain
import numpy as np
from numpy import matrix, linalg, pi
import math
from scipy.stats import multivariate_normal

from oogan_modules import Flatten, UpConvBlock, DownConvBlock, CodeReduction, InfoGANInput, OOGANInput

from numbers import Number
from dist import Normal
from copy import deepcopy


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

dist = Normal()
def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)



  
ngf_multi_dic = {32:[4,4,2,1],
                 64:[8,4,4,2,1],
                 128:[8,8,8,4,4,2],
                 256:[16,8,8,8,4,4,2],}
class Generator(nn.Module):
    def __init__(self, ngf=64, c_dim=16, z_dim=0, im_size=64, nc=1, input_module='OOGAN'):
        super(Generator, self).__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        
        ngf_multi = ngf_multi_dic[im_size]
        
        self.init_block = OOGANInput(c_dim, z_dim, ngf_multi[0]*ngf, ngf_multi[1]*ngf)
        if not input_module == "OOGAN":
            self.init_block = InfoGANInput(c_dim, z_dim, ngf_multi[0]*ngf, ngf_multi[1]*ngf)

        self.blocks = [] 
        for i in range(len(ngf_multi)-2):
            self.blocks.append(
                UpConvBlock(ngf*ngf_multi[i+1], ngf*ngf_multi[i+2]))
        
        self.blocks.append(
            nn.Sequential(
                spectral_norm(
                    nn.Conv2d(ngf*ngf_multi[-1], nc, 1, 1, 0, bias=True)), 
                nn.Tanh()))
    
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, c, z=None):
        feat = self.init_block(c=c, z=z)
        return self.blocks(feat)


ndf_multi_dic = {32:[1,2,4,4],
                 64:[1,2,4,4,8],
                 128:[1,2,4,8,8,16],
                 256:[1,2,4,8,8,16,16],}
class Discriminator(nn.Module):
    def __init__(self, ndf=64, c_dim=16, im_size=64, nc=1, prob_c=False, output_module="OOGAN"):
        super(Discriminator, self).__init__()

        feat_16 = [spectral_norm(nn.Conv2d(nc, ndf, 1, 1, 0, bias=True))]
        ndf_multi = ndf_multi_dic[im_size]
        for i in range(len(ndf_multi)-3):
            feat_16.append(
                DownConvBlock(ndf*ndf_multi[i], ndf*ndf_multi[i+1]))
        self.feat_16 = nn.Sequential(*feat_16)
        
        self.real_fake = nn.Sequential(
            DownConvBlock(ndf*ndf_multi[-3], ndf*ndf_multi[-2]),
            DownConvBlock(ndf*ndf_multi[-2], ndf*ndf_multi[-1]),
            spectral_norm(nn.Conv2d(ndf*ndf_multi[-1], 1, 4, 1, 0, bias=True)))
        
        self.pred_c = nn.Sequential( 
            DownConvBlock(ndf*ndf_multi[-3], ndf*ndf_multi[-2]),
            CodeReduction(c_dim, ndf*ndf_multi[-2], 8, prob=prob_c))
        if not output_module == "OOGAN":
            if prob_c:
                c_dim = c_dim*2
            self.pred_c = nn.Sequential(
                DownConvBlock(ndf*ndf_multi[-3], ndf*ndf_multi[-2]),
                DownConvBlock(ndf*ndf_multi[-2], c_dim),
                spectral_norm(nn.Conv2d(c_dim, c_dim, 4, 1, 0, bias=True)),
                Flatten())

    def forward(self, img):
        feat_16 = self.feat_16(img)
        return self.real_fake(feat_16).view(-1), self.pred_c(feat_16)

def get_item(pred):
    return pred.mean().item()

class DisentangleGAN(nn.Module):
    def __init__(self, device, ngf, ndf, z_dim, c_dim, im_size, nc, g_type, d_type, prob_c, one_hot, lr, recon_weight, onehot_weight):
        super().__init__()
        
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.device = device
        self.prob_c = prob_c
        self.one_hot = one_hot
        self.g_type = g_type
        self.d_type = d_type

        self.recon_weight = recon_weight
        self.onehot_weight = onehot_weight

        self.generator = Generator(ngf=ngf, c_dim=c_dim, z_dim=z_dim, im_size=im_size, nc=nc, input_module=g_type).to(device)
        self.discriminator = Discriminator(ndf=ndf, c_dim=c_dim, im_size=im_size, nc=nc, prob_c=prob_c, output_module=d_type).to(device)

        self.opt_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.99))
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.99))
        self.opt_info = optim.Adam(chain(self.generator.parameters(), self.discriminator.pred_c.parameters()), lr=lr, betas=(0.5, 0.99))

        self.running_avg_g = None 
        
        self.real_images = None

    def save(self, path):
        backup_para = copy_G_params(self.generator)
        load_params(self.generator, self.running_avg_g)
        torch.save(self.state_dict(), path)
        load_params(self.generator, backup_para)

    def generate_random_sample(self, save_path, z=None, c=None, batch_size=16):
        backup_para = copy_G_params(self.generator)
        load_params(self.generator, self.running_avg_g)
        #self.generator.eval()
        if self.z_dim > 0 and z is None:
            z = torch.randn(batch_size, self.z_dim).to(self.device)
        if self.c_dim > 0 and c is None:
            c = torch.randn(batch_size, self.c_dim).uniform_(0,1).to(self.device)
            #c = torch.randn(1, self.c_dim).uniform_(0,1).repeat(batch_size,1).to(self.device)
        with torch.no_grad():
            g_img = self.generator(c=c, z=z).cpu()
            vutils.save_image(g_img.add_(1).mul_(0.5), save_path.replace(".jpg", "_random.jpg"), pad_value=0)
            del g_img
        #self.generator.train()
        load_params(self.generator, backup_para)

    def generate_each_dim(self, save_path, dim=0, z=None, c=None, num_interpolate=10, num_samples=8):
        #self.generator.eval()
        if self.running_avg_g is not None:
            backup_para = copy_G_params(self.generator)
            load_params(self.generator, self.running_avg_g)

        if self.z_dim > 0 and z is None:
            z = torch.randn(num_samples, self.z_dim).to(self.device)
            z = z.unsqueeze(1).repeat(1, num_interpolate, 1).view(-1, self.z_dim)
        if self.c_dim > 0 and c is None:
            c = torch.randn(num_samples, self.c_dim).uniform_(0.2, 0.6).to(self.device)
            c = c.unsqueeze(1).repeat(1, num_interpolate, 1)   #.view(-1, self.z_dim)
            c_line = torch.linspace(0, 1, num_interpolate).to(self.device)
            c_line = c_line.unsqueeze(0).repeat(num_samples, 1)
            c[:,:,dim] = c_line
            c = c.view(-1, self.c_dim)
        with torch.no_grad():
            g_img = self.generator(c=c, z=z)
            vutils.save_image(g_img.add_(1).mul_(0.5), \
                save_path.replace(".jpg", "_dim_%d.jpg"%(dim)), pad_value=0,nrow=num_interpolate)
            del g_img
        if self.running_avg_g is not None:
            load_params(self.generator, backup_para)

    def neg_log_density(self, sample, params):
        constant = torch.Tensor([np.log(2 * np.pi)]).to(self.device)
        mu = params[:,:self.c_dim]
        logsigma = params[:,self.c_dim:]
        inv_sigma = torch.exp(-logsigma)
        tmp = (sample - mu) * inv_sigma
        return 0.5 * (tmp * tmp + 2 * logsigma + constant)

    def sample_hot_c(self, batch_size, c_dim, num_hot=1):
        y_onehot = torch.zeros(batch_size, c_dim)
        if num_hot==1:
            y = torch.LongTensor(batch_size,1).random_() % c_dim
            y_onehot.scatter_(1, y, 1)
            return y_onehot.to(self.device), y.view(-1).to(self.device)
        else:
            for _ in range(num_hot):
                y = torch.LongTensor(batch_size,1).random_() % c_dim
                y_onehot.scatter_(1, y, 1)
        return y_onehot.to(self.device)

    def sample_z_and_c(self, batch_size, n_iter):
        # sample z from Normal distribution
        z = None
        if self.z_dim > 0:
            z = torch.randn(batch_size, self.z_dim).to(self.device)
        
        # sample c alternativaly from uniform and onehot
        c_idx = None
        if n_iter%4==0 and self.one_hot:
            c = torch.Tensor(batch_size, self.c_dim).uniform_(0.2,0.6).to(self.device)
            choosen_dim = np.random.randint(0, self.c_dim)
            c[:, choosen_dim] = 1
            c_idx = torch.Tensor(batch_size).fill_(choosen_dim).long().to(self.device)
        elif n_iter%2==0 and self.one_hot:
            c, c_idx = self.sample_hot_c(batch_size, c_dim=self.c_dim, num_hot=1)
        else:
            c = torch.Tensor(batch_size, self.c_dim).uniform_(0, 1).to(self.device)
        return z, c, c_idx

    def compute_gradient_penalty(self, real_images, fake_images):
        # Compute gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1).expand_as(real_images).to(self.device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images).clone().detach().requires_grad_(True)
        
        out = self.discriminator(interpolated)[0]
        
        exp_grad = torch.ones(out.size()).to(self.device)
        grad = torch.autograd.grad(outputs=out,
                                    inputs=interpolated,
                                    grad_outputs=exp_grad,
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
        return d_loss_gp

    def compute_total_correlation(self):
        real_images = torch.cat(self.real_images, dim=0)
        
        batch_size = real_images.size(0)
        print(batch_size)
        self.discriminator.eval()
        with torch.no_grad():
            c_params = self.discriminator(real_images)[1]
        self.discriminator.train()

        sample_c = dist.sample(params=c_params.view(batch_size, self.c_dim, 2))
        _logqc = dist.log_density( sample_c.view(-1, 1, self.c_dim), c_params.view(1, -1, self.c_dim, 2) )

        logqc_prodmarginals = (logsumexp(_logqc, dim=1, keepdim=False) - math.log(batch_size)).sum(1)
        logqc = (logsumexp(_logqc.sum(2), dim=1, keepdim=False) - math.log(batch_size))
        
        #print( logqc, logqc_prodmarginals )
        self.real_images = None
        return (logqc - logqc_prodmarginals).mean().item()


    def train(self, real_image, n_iter):
        if self.running_avg_g is None:
            self.running_avg_g = copy_G_params(self.generator)

        batch_size = real_image.size(0)

        ### prepare data part
        z, c, c_idx = self.sample_z_and_c(batch_size, n_iter)
        g_img = self.generator(c=c, z=z)        
        r_img = real_image.to(self.device)

        ### Discriminator part
        self.discriminator.zero_grad()
        pred_r, _ = self.discriminator(r_img)
        pred_f, _ = self.discriminator(g_img.detach())
        loss_d = F.relu(1-pred_r).mean() + F.relu(1+pred_f).mean()
        loss_d.backward()
        loss_d_gp = 10 * self.compute_gradient_penalty(r_img, g_img.detach())
        loss_d_gp.backward()
        self.opt_disc.step()

        ### Generator part
        self.generator.zero_grad()
        pred_g, _ = self.discriminator(g_img)
        loss_g = -pred_g.mean()
        loss_g.backward()
        self.opt_gen.step()

        ### Mutual Information between c and c' Part
        self.generator.zero_grad()
        self.discriminator.zero_grad()

        z, c, c_idx = self.sample_z_and_c(batch_size, n_iter)
        g_img = self.generator(c=c, z=z)        
        pred_g, pred_c_params = self.discriminator(g_img)

        if self.prob_c:
            loss_g_recon_c = self.neg_log_density(c, pred_c_params).mean()
        else:
            loss_g_recon_c = F.l1_loss(pred_c_params, c)

        loss_g_onehot = torch.Tensor([0]).to(self.device)
        if n_iter%4==0 and self.one_hot:
            loss_g_onehot = 0.2*F.cross_entropy(pred_c_params[:,:self.c_dim], c_idx)
        elif n_iter%2==0 and self.one_hot:
            loss_g_onehot = 0.8*F.cross_entropy(pred_c_params[:,:self.c_dim], c_idx)
           
        loss_info = self.recon_weight * loss_g_recon_c + self.onehot_weight * loss_g_onehot
        loss_info.backward()

        self.opt_info.step()

        for p, avg_p in zip(self.generator.parameters(), self.running_avg_g):
            avg_p.mul_(0.999).add_(0.001, p.data)

        return get_item(pred_r), get_item(pred_f), get_item(pred_g), loss_g_onehot.item()*2, loss_g_recon_c.item()
    
