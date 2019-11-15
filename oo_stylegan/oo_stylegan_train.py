from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from oo_stylegan_modules import StyledGenerator, Discriminator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def lsun_loader(path):
    def loader(transform):
        data = datasets.LSUNClass(
            path, transform=transform,
            target_transform=lambda x: 0)
        data_loader = DataLoader(data, shuffle=False, batch_size=batch_size,
                                 num_workers=4)

        return data_loader

    return loader


def celeba_loader(path):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                 num_workers=4)

        return data_loader

    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform)

    return loader

    # for img, label in loader:
    #     yield img, label
import numpy as np
normalization = torch.Tensor([np.log(2 * np.pi)])
def NLL(sample, params):
    """Analytically computes
       E_N(mu_2,sigma_2^2) [ - log N(mu_1, sigma_1^2) ]
       If mu_2, and sigma_2^2 are not provided, defaults to entropy.
    """
    mu = params[:,:,0]
    logsigma = params[:,:,1]
        
    c = normalization.to(mu.device)
    inv_sigma = torch.exp(-logsigma)
    tmp = (sample - mu) * inv_sigma
    return torch.mean(0.5 * (tmp * tmp + 2 * logsigma + c))


def sample_onehot(batch_size, c_dim, device):
    # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
    y = torch.LongTensor(batch_size,1).random_() % c_dim
    # One hot encoding buffer that you create out of the loop and just keep reusing
    y_onehot = torch.FloatTensor(batch_size, c_dim)

    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot.to(device), y.view(-1).to(device)


def interpolate(c_dim, device):
    b_size = 8*c_dim
    inter = torch.linspace(-1.2, 1.2, 8)
    c = torch.Tensor(1, c_dim).uniform_(0.2, 0.5).expand(b_size, c_dim).contiguous()
    for i in range(c_dim):
        c[i*8:(i+1)*8,i] = inter
    return c.to(device)

def train(generator, discriminator, init_step, loader, options):
    step = init_step # can be 1,2,3,4,5,6
    data_loader = sample_data(loader, 4 * 2 ** step)
    dataset = iter(data_loader)

    total_iter = 600000
    total_iter_remain = total_iter - (total_iter//6)*(step-1)

    pbar = tqdm(range(total_iter_remain))

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0
    info_loss_nll = 0
    info_loss_onehot = 0

    from datetime import datetime
    import os
    date_time = datetime.now()
    post_fix = '%s_%d_%d.txt'%(date_time.date(), date_time.hour, date_time.minute)
    log_folder = 'trial_%s_%d_%d'%(date_time.date(), date_time.hour, date_time.minute)
    
    os.mkdir(log_folder)
    os.mkdir(log_folder+'/checkpoint')
    os.mkdir(log_folder+'/sample')

    log_file_name = os.path.join(log_folder, 'train_log_'+post_fix)
    log_file = open(log_file_name, 'w')
    log_file.write('g,d,nll,onehot\n')
    log_file.close()

    from shutil import copy
    copy('oo_stylegan_train.py', log_folder+'/train_%s.py'%post_fix)
    copy('oo_stylegan_modules.py', log_folder+'/model_%s.py'%post_fix)

    alpha = 0
    one = torch.FloatTensor([1]).cuda()
    mone = one * -1
    iteration = 0

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, (2/(total_iter//6)) * iteration)

        if iteration > total_iter//6:
            alpha = 0
            iteration = 0
            step += 1

            if step > 5:
                alpha = 1
                step = 5
            data_loader = sample_data(loader, 4 * 2 ** step)
            dataset = iter(data_loader)

        try:
            real_image, label = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, label = next(dataset)

        iteration += 1

        b_size = real_image.size(0)
        real_image = real_image.cuda()
        label = label.cuda()
        real_predict, _ = discriminator(
            real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() \
            - 0.001 * (real_predict ** 2).mean()
        real_predict.backward(mone)

        # sample input data for Generator
        gen_in = torch.randn(b_size, code_size, device='cuda')

        # sample control vector c
        gen_c = torch.Tensor(b_size, C_DIM).uniform_(0, 1).cuda()
        if (i + 1) % (n_critic*2) == 0:
            gen_c, gen_c_idx = sample_onehot(real_image.size(0), C_DIM, device='cuda')
            #tos = np.random.randint(0,2)
            #if tos==0:
            #    gen_c = -gen_c

        # train Discriminator
        fake_image = generator(
            gen_in, gen_c,
            step=step, alpha=alpha)
        fake_predict, _ = discriminator(
            fake_image.detach(), step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        eps = torch.rand(b_size, 1, 1, 1).cuda()
        x_hat = eps * real_image.data + (1 - eps) * fake_image.detach().data
        x_hat.requires_grad = True
        hat_predict, _ = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
                         .norm(2, dim=1) - 1)**2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val += grad_penalty.item()
        disc_loss_val += (real_predict - fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()
            
            predict, pred_c_params = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()
            gen_loss_val += loss.item()


            loss_nll = NLL(gen_c, pred_c_params)
            info_loss_nll += loss_nll.item()
            if loss_nll.item()<100:
                ## we find that during the upscale moment, the loss can sometimes increase dramatically
                ## should ignore that period and wait generator become stable before keep training on mutual information
                loss = loss + loss_nll

            if (i + 1) % (n_critic*2) == 0:
                loss_onehot = F.cross_entropy(pred_c_params[:,:,0], gen_c_idx)
                #if tos==0:
                #    loss_onehot = F.cross_entropy(-pred_c_params[:,:,0], gen_c_idx)
                #else:
                #    loss_onehot = F.cross_entropy(pred_c_params[:,:,0], gen_c_idx)
                
                info_loss_onehot += loss_onehot.item()
                if loss_onehot.item()<10:
                    loss = loss + loss_onehot

            loss.backward()
            info_optimizer.step()
            accumulate(g_running, generator)

        if (i + 1) % 1000 == 0 or i==0:
            with torch.no_grad():
                images = g_running(torch.randn(5 * 10, code_size).cuda(),
                                torch.Tensor(5*10, C_DIM).uniform_(0,1).cuda(),
                                    step=step, alpha=alpha).data.cpu()

                utils.save_image(
                    images,
                    f'{log_folder}/sample/{str(i + 1).zfill(6)}.png',
                    nrow=10,
                    normalize=True,
                    range=(-1, 1))

                inter_c = interpolate(C_DIM, real_image.device)
                z = torch.randn(1, code_size).expand(inter_c.size(0), code_size).contiguous().to(inter_c.device)
                
                images = g_running(z, inter_c, fix_noise=True, step=step, alpha=alpha).data.cpu()

                utils.save_image(
                    images,
                    f'{log_folder}/sample/pattern_{str(i + 1).zfill(6)}.png',
                    nrow=8,
                    normalize=True,
                    range=(-1, 1))
                print('pattehn')

        if (i+1) % 10000 == 0 or i==0:
            try:
                torch.save(g_running.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_g.model')
                torch.save(discriminator.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_d.model')
            except:
                pass

        if (i+1)%50 == 0:
            
            state_msg = (f'{i + 1}; G: {gen_loss_val/(50//n_critic):.3f}; D: {disc_loss_val/50:.3f};'
                f' Info_nll: {info_loss_nll/(50//n_critic):.3f}; Info_onehot: {info_loss_onehot/(50//(n_critic*2)):.3f};'
                f' Grad: {grad_loss_val/50:.3f}; Alpha: {alpha:.3f}')
            

            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f,%.5f,%.5f\n"%(gen_loss_val/(50//n_critic), disc_loss_val/50, \
                info_loss_nll/(50//(n_critic*2)),info_loss_onehot/(50//(n_critic*4)))
            log_file.write(new_line)
            log_file.close()

            disc_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0
            info_loss_nll = 0
            info_loss_onehot = 0
            print(state_msg)

            #pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 4
    n_critic = 1

    parser = argparse.ArgumentParser(description='StyleGANs of the OOGAN design for better disentangling')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--init-size', default=8, type=int, help='initial image size')
    parser.add_argument('--mixing', action='store_true', help='use mixing regularization')
    parser.add_argument('-d', '--data', default='celeba', type=str,
                        choices=['celeba', 'lsun'],
                        help=('Specify dataset. '
                            'Currently CelebA and LSUN is supported'))

    args = parser.parse_args()

    C_DIM = 16
    generator = StyledGenerator(code_size, control_dim=C_DIM).cuda()
    discriminator = Discriminator(control_dim=C_DIM).cuda()
    g_running = StyledGenerator(code_size, control_dim=C_DIM).cuda()
    
    #generator.load_state_dict(torch.load('checkpoint/150000_g.model'))
    #g_running.load_state_dict(torch.load('checkpoint/150000_g.model'))
    #discriminator.load_state_dict(torch.load('checkpoint/150000_d.model'))
    
    g_running.train(False)

    class_loss = nn.CrossEntropyLoss()

    g_optimizer = optim.Adam(generator.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({'params': generator.style.parameters(), 'lr': args.lr * 0.01})
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    from itertools import chain
    info_optimizer = optim.Adam(chain(generator.generator.parameters(), discriminator.to_c.parameters()), lr=args.lr, betas=(0.0, 0.99))
    info_optimizer.add_param_group({'params': generator.style.parameters(), 'lr': args.lr * 0.01})
    
    accumulate(g_running, generator, 0)

    if args.data == 'celeba':
        loader = celeba_loader(args.path)

    elif args.data == 'lsun':
        loader = lsun_loader(args.path)

    init_step = 1
    train(generator, discriminator, init_step, loader, args)
