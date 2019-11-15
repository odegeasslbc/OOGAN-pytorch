from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.optim as optim
from torchvision.models import vgg
from tqdm import tqdm
from torchvision import utils as vutils
from torch.nn import functional as F 
import numpy as np

class VGG_celeba(nn.Module):
    def __init__(self, nclass=40):
        super(VGG_celeba, self).__init__()
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, nclass))

    def load_pretrain_weights(self):
        pretrained_vgg16 = vgg.vgg16(pretrained=True)
        self.features.load_state_dict(pretrained_vgg16.features.state_dict())
        self.classifier[0] = pretrained_vgg16.classifier[0] 
        self.classifier[3] = pretrained_vgg16.classifier[3] 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def make_layers(self, cfg="D", batch_norm=False):
        cfg_dic = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        cfg = cfg_dic[cfg]
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, img):
        pred = self.classifier( self.avgpool(self.features(img)).view(img.size(0), -1) )
        return pred

    def get_feat(self, img):
        return self.features(img)



from oogan_models import DisentangleGAN
import pickle

config = pickle.load(open("/path/to/saved/config.pkl", "rb"))

device = torch.device("cuda:%d"%(config["CUDA_ID"]))
Z_DIM = config["Z_DIM"]
C_DIM = config["C_DIM"]

net = DisentangleGAN(device=device, ngf=config["NGF"], ndf=config["NDF"], z_dim=config["Z_DIM"], c_dim=config["C_DIM"], \
        im_size=config["IM_SIZE"], nc=config["NC"], g_type=config["G_TYPE"], d_type=config["D_TYPE"], prob_c=config["USE_PROB_C"], \
        lr=config["LR"], one_hot=config["ONE_HOT"], recon_weight=config["LAMBDA"], onehot_weight=config["GAMMA"])
net.load_state_dict(torch.load('/path/to/model.pth'))
g = net.generator.eval()

vgg = VGG_celeba()
vgg.load_state_dict(torch.load('./model.pth'))
vgg.to(device)
vgg.eval()

for p in g.parameters():
    p.requires_grad = False
for p in vgg.parameters():
    p.requires_grad = False


def get_pairwise_difference(batch_size=100, trial_num=1000):
    total_diff = []
    for it in range(trial_num):
        z = torch.randn(batch_size, Z_DIM).to(device)
        c = torch.randn(batch_size, C_DIM).uniform_(-1, 1).to(device)
            
        dim_a = np.random.randint(0, C_DIM, size=batch_size)
        dim_b = (dim_a + np.random.randint(1, C_DIM, size=batch_size)) % C_DIM

        c1 = c.clone()
        c2 = c.clone()
        for i in range(batch_size):
            c1[i,dim_a[i]] = -1
            c1[i,dim_b[i]] = 1

            c2[i,dim_a[i]] = 1
            c2[i,dim_b[i]] = -1

        with torch.no_grad():
            gimg1 = g(z=z, c=c1)
            gimg2 = g(z=z, c=c2)

            diff = F.l1_loss( vgg.get_feat(gimg1), vgg.get_feat(gimg2) ).item()
            total_diff.append(diff)
    print( "the average score is: %.5f    the std of the score is: %.5f"%(np.mean(total_diff), np.std(total_diff)) )
    return total_diff


if __name__ == "__main__":
    get_pairwise_difference()
