import torch
import torchvision.datasets as Dataset
from torch.utils.data import DataLoader
from helper_functions import InfiniteSamplerWrapper, make_folders, trans_maker
from os.path import join as pjoin
import tqdm

from oogan_models import DisentangleGAN
from config import training_config as config


def train(net, dataloader):
    TRIAL_NAME = config["TRIAL_NAME"]
    MAX_ITERATION = config["MAX_ITERATION"]
    LOG_INTERVAL = MAX_ITERATION//2000
    SAVE_IMAGE_INTERVAL = MAX_ITERATION//500
    SAVE_MODEL_INTERVAL = MAX_ITERATION//50

    # prepare variables for logging purpose
    D_real = D_fake = G_real = C_hot = C_recon = 0
    # prepare paths to save models and images generated during training
    saved_image_folder, saved_model_folder = make_folders(config["SAVE_FOLDER"], TRIAL_NAME, config)

    for n_iter in tqdm.tqdm(range(0, MAX_ITERATION+1)):
        if n_iter % SAVE_IMAGE_INTERVAL == 0:
            net.generate_random_sample(save_path=pjoin(saved_image_folder, "random_%d.jpg"%n_iter))
        if n_iter % SAVE_MODEL_INTERVAL == 0:
            #torch.save(net.state_dict(), pjoin(saved_model_folder, "%d.pth"%n_iter))    
            net.save( pjoin(saved_model_folder, "%d.pth"%n_iter) )
            
        ### 0. prepare data
        real_image = next(dataloader)
        if type(real_image) is list:
            real_image = real_image[0]

        # noise annealing trick on real images for better GAN convergence
        if config["NOISE_ANNEL"]:
            threshold = MAX_ITERATION//10
            if n_iter < threshold:
                noise = torch.Tensor(real_image.shape).normal_(0, ((threshold - n_iter) / threshold )).to(real_image.device)
                real_image = real_image + noise

        ### 1. Train the model
        loss_r, loss_f, loss_g, loss_onehot, loss_recon = net.train(real_image, n_iter)

        ### 2. display training log
        D_fake += loss_f
        D_real += loss_r
        G_real += loss_g
        C_hot += loss_onehot
        C_recon += loss_recon

        if n_iter % LOG_INTERVAL == 0 and n_iter > 0:
            print("\n%s\n: D(x): %.5f    D(G(z)): %.5f    G(z): %.5f    C_onehot: %.5f    C_recon: %.5f"%\
                    (TRIAL_NAME, D_real/LOG_INTERVAL, D_fake/LOG_INTERVAL, G_real/LOG_INTERVAL, C_hot/LOG_INTERVAL, C_recon/LOG_INTERVAL))
            D_real = D_fake = G_real = C_hot = C_recon = 0



if __name__ == "__main__":
    ### prepare the dataloader
    device = torch.device("cuda:%d"%(config["CUDA_ID"]))

    if "dsprite" not in config["DATASET"]:
        dataset = Dataset.ImageFolder(root=config["DATA_ROOT"], transform=trans_maker(config["IM_SIZE"])) 
    else:
        from helper_functions import DSpriteDataset
        dataset = DSpriteDataset(config["DATA_ROOT"], im_size=config["IM_SIZE"])

    dataloader = iter(DataLoader(dataset, config["BATCH_SIZE"], \
            sampler=InfiniteSamplerWrapper(dataset), num_workers=config["DATALOADER_WORKERS"], pin_memory=True))

    ### init the network
    net = DisentangleGAN(device=device, ngf=config["NGF"], ndf=config["NDF"], z_dim=config["Z_DIM"], c_dim=config["C_DIM"], \
        im_size=config["IM_SIZE"], nc=config["NC"], g_type=config["G_TYPE"], d_type=config["D_TYPE"], prob_c=config["USE_PROB_C"], \
        lr=config["LR"], one_hot=config["ONE_HOT"], recon_weight=config["LAMBDA"], onehot_weight=config["GAMMA"])

    ### train the model
    train(net, dataloader)