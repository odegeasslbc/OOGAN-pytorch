from torchvision import transforms
import torch.utils.data as data
import torch
import torch.nn.functional as F

import numpy as np
import os
from shutil import copyfile
import pickle



class DSpriteDataset(data.Dataset):
    def __init__(self, data_root='data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', im_size=64):
        super(DSpriteDataset, self).__init__()
        self.frame = np.load(data_root)['imgs']
        self.im_size = im_size
        
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        result = self.frame[idx]
        image = torch.Tensor(result).unsqueeze(0).unsqueeze(0).mul_(2).add_(-1)
        if self.im_size == 32:
            image = F.interpolate(image, scale_factor=0.5)
        return image.view(1, self.im_size, self.im_size)

def _rescale(img):
    return img * 2.0 - 1.0

def trans_maker(size=256):
    trans = transforms.Compose([
                    transforms.Resize((size+int(size*0.15), size+int(size*0.15))),
                    transforms.CenterCrop((size, size)), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    _rescale
                    ])
    return trans

# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L5-L15
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L18-L26
class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def make_folders(save_folder, trial_name, config_to_save=None):
    saved_model_folder = os.path.join(save_folder, 'train_results/%s/models'%trial_name)
    saved_image_folder = os.path.join(save_folder, 'train_results/%s/images'%trial_name)
    folders = [os.path.join(save_folder, 'train_results'), os.path.join(save_folder, 'train_results/%s'%trial_name), 
    os.path.join(save_folder, 'train_results/%s/images'%trial_name), os.path.join(save_folder, 'train_results/%s/models'%trial_name)]
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    if config_to_save is not None:
        pickle.dump(config_to_save, open(saved_image_folder+"/../config.pkl", 'wb'))
    return saved_image_folder, saved_model_folder 




