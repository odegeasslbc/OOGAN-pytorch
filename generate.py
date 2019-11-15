import torch
import torchvision.datasets as Dataset
from torch.utils.data import DataLoader
from helper_functions import InfiniteSamplerWrapper, make_folders, trans_maker
from os.path import join as pjoin
import tqdm

from oogan_models import DisentangleGAN
from config import training_config as config


device = torch.device("cuda:%d"%(config["CUDA_ID"]))

### init the network
net = DisentangleGAN(device=device, ngf=config["NGF"], ndf=config["NDF"], z_dim=config["Z_DIM"], c_dim=config["C_DIM"], \
        im_size=config["IM_SIZE"], nc=config["NC"], g_type=config["G_TYPE"], d_type=config["D_TYPE"], prob_c=config["USE_PROB_C"], \
        lr=config["LR"], one_hot=config["ONE_HOT"], recon_weight=config["LAMBDA"], onehot_weight=config["GAMMA"])


net.load_state_dict(torch.load('./train_results/trial_celeba_oogan_2/models/30000.pth'))


for dim in range(net.c_dim):
    try:
        net.generate_each_dim('text.jpg', dim=dim, num_interpolate=10, num_samples=4)
    except:
        print("error happens when generating the images on dim %d"%dim)