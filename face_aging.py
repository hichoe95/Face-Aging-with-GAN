
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from __future__ import print_function
import os.path
import torch.nn.parallel
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import PIL
import random
from torchvision.utils import save_image

torch.cuda.set_device(2)
torch.cuda.current_device()


# In[5]:


class Data_Loader(data.Dataset):
    def __init__(self, img_dir, transform, label, mode):
        self.img_dir = img_dir
        self.transform = transform
        self.label = label
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()
        
        if mode == 'train':
            self.num_imgs = len(self.train_dataset)
        else:
            self.num_imgs = len(self.test_dataset)
    
    def preprocess(self):
        file_list = os.listdir(self.img_dir)
        
        random.seed(1234)
        
        for i in range(len(file_list)*80):
            
            if (i+1)< 500:
                self.test_dataset.append([random.choice(file_list), self.label])
                #print(self.label)
            else:
                self.train_dataset.append([random.choice(file_list), self.label])
            
        print('Finished preprocessing...')
        
    def __getitem__(self,index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        #print(label)
        image = Image.open(os.path.join(self.img_dir, filename))
        return self.transform(image), label
        
    def __len__(self):
        return self.num_imgs


# In[6]:


def get_loader(img_dir, label, batch_size = 16, mode='train', num_workers=1):
    transform = []
    
#     transform.append(transforms.CenterCrop(128))
#     transform.append(transforms.Resize(64))
    transform.append(transforms.ToTensor())
#     transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)
    
    dataset = Data_Loader(img_dir = img_dir , transform = transform , label = label, mode = mode)
    
    data_loader = data.DataLoader(dataset = dataset,
                                  batch_size = batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader


# In[7]:


# imgdir_128_under30 = os.path.join('../../data/only_face', '25')
# imgdir_128_over40 = os.path.join('../../data/only_face', '50')
# imgdir_64_under30 = os.path.join('../../data/Datas', '64_under_30')
# imgdir_64_over40 = os.path.join('../../data/Datas', '64_over_40')


# under30 = get_loader(img_dir = imgdir_128_under30, label = 1, batch_size = 8)
# over40 = get_loader(img_dir = imgdir_128_over40, label = 0, batch_size = 8)


# In[8]:


# def imshow(inp):
#     inp = inp.numpy().transpose((1,2,0))
#     #print(inp.shape)
#     inp = np.clip(inp, 0, 1) 
#     plt.imshow(inp)

# def gray_imshow(inp):
#     inp = inp.numpy()#.transpose((1,2,0))
#     print(inp.shape)
#     plt.imshow(inp,cmap = plt.get_cmap('gray'))
    
    
# for i,k in enumerate(under30):
#     img, label = k
#     if i == 100:
#         print(img)
#     if i % 1000 == 999:
#         grays = torch.from_numpy(np.resize(img.numpy(), (4, 1, 128, 128)))
#         out = torchvision.utils.make_grid(grays)
#         print(i)
#         print(label)
#         #print(label.numpy())
#         imshow(out)
#         plt.show()

# for i,k in enumerate(over40):
#     img, label = k
#     if i == 100:
#         print(img)
#     if i % 1000 == 999:
#         grays = torch.from_numpy(np.resize(img.numpy(), (4, 1, 128, 128)))
#         out = torchvision.utils.make_grid(grays)
#         print(i)
#         print(label)
#         #print(label.numpy())
#         imshow(out)
#         plt.show()


# In[9]:


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias = False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace = True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True)
        )
        
    def forward(self, x):
        return x+self.main(x)


# In[10]:


class Generator(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=6, n_dim = 1):
        super(Generator, self).__init__()
        
        layers = []
        layers.append(nn.Conv2d(1+n_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        
        # endoing layers
        curr_dim = conv_dim
        
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        
        # bottleneck layers
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        
        # decoding layers
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim//2
            
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))

        layers.append(nn.Sigmoid()) # output is gray scale image
        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.main(x)

        return out


# In[11]:


class Discriminator(nn.Module):
    def __init__(self, batch_size, image_size=128, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding = 1))
        layers.append(nn.LeakyReLU(0.01))
        
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim*2
        
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size = 3, stride=1, padding=1, bias=False)
        
#         self.sig = nn.Sigmoid()
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        
        return out_src


# In[15]:


import time
import datetime

class Solver(object):
    
    def __init__(self, young, old, batch_size = 4, image_size = 128, g_lr = 0.0001, d_lr = 0.0001, num_iters = 30):
        
        self.young = young
        self.old = old
        self.batch_size = batch_size
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.num_iters = num_iters
        self.image_size = image_size
        self.g_conv_dim = 64
        self.d_conv_dim = 64
        self.repeat_num = 2
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.build_model()
        
        
    def build_model(self):
        self.G = Generator(self.g_conv_dim, self.repeat_num, n_dim = 1)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.repeat_num)
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        
        self.criterion = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        
#         self.G.apply(self.weights_init_normal)
#         self.D.apply(self.weights_init_normal)
        self.G.to(self.device)
        self.D.to(self.device)

        
    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        
    def print_img(self,inp):
        inp = torch.from_numpy(np.resize(inp.cpu().detach().numpy(), (self.batch_size, 1, 128, 128)))
        inp = torchvision.utils.make_grid(inp)
        inp = inp.numpy().transpose((1,2,0))
        inp = np.clip(inp, 0, 1) 
        plt.imshow(inp)
    
    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
    
    def create_labels(self):
#         target_domain = torch.zeros([self.batch_size, 1, self.image_size, self.image_size]).to(self.device)
        target_domain = np.random.rand(self.batch_size, 1, self.image_size, self.image_size)
        target_domain = torch.FloatTensor(np.rint(target_domain)).to(self.device)
        
        recons_domain = np.random.rand(self.batch_size, 1, self.image_size, self.image_size)
        recons_domain = torch.FloatTensor(np.rint(recons_domain)).to(self.device)
        
        return target_domain/5, recons_domain/5
    
    def denorm(self, x):
        out = (x+1)/2
        return out.clamp_(0,1)
    
    def weights_init_normal(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant(m.bias.data, 0.0)
        
    def train(self):
        
        # learning rate cache for decaying
        young = self.young
        old = self.old
        
        print('young len : {}'.format(young.__len__()))
        print('old len : {}'.format(old.__len__()))
        
        g_lr = self.g_lr
        d_lr = self.d_lr
        
        # start training
        

        young_iter = iter(young)
        old_iter = iter(old)

        # fetch fixed inputs for debugging
        young_fixed, young_label = next(young_iter)
        old_fixed, old_label = next(old_iter)
        
        print(young_label)
        
        start_time = time.time()
        
        print('start training...')
        
        iteration = min(old.__len__(), young.__len__())
        
        t_domain, r_domain = self.create_labels()
        print(young_fixed.size(0))
        correct = 0
        for i in range(iteration-1):

            # real image and fake image
            young_face, young_label = next(young_iter)
            old_face, old_label = next(old_iter)

            if (old_face.size(0) != self.batch_size or young_face.size(0) != self.batch_size or old_face.size(2) != self.image_size or young_face.size(2) != self.image_size) :
                continue

            young_face = young_face.to(self.device)
            old_face = old_face.to(self.device)
            young_label = young_label.float().view(self.batch_size, 1).to(self.device)
            old_label = old_label.float().view(self.batch_size, 1).to(self.device)
            
            ################################## Train Discriminator ##################################
            #                                                                                       #
            #########################################################################################
            
            # compute loss with real image
            out_src = self.D(old_face)

            d_loss_real = - torch.mean(out_src)
#             d_loss_real = torch.mean((out_src-1)**2)   
#             d_loss_real = self.MSE(out_src,old_label)
    
            # compute loss with fake image
            # concatenate fake image with target domain
            young_input = torch.cat([young_face, t_domain] , dim = 1)
            x_fake = self.G(young_input)

            out_src = self.D(x_fake.detach())

            d_loss_fake = torch.mean(out_src)
#             d_loss_fake = torch.mean((out_src)**2)
#             d_loss_fake = self.MSE(out_src, young_label)
    
#             # compute gradient penalty
#             alpha = torch.rand(old_face.size(0), 1, 1, 1).to(self.device)
#             x_hat = (alpha * old_face.data + (1 - alpha) * young_face.data).requires_grad_(True)
#             out_src = self.D(x_hat)
#             d_loss_gp = self.gradient_penalty(out_src, x_hat)
            

            # backward and optimize
            d_loss = (d_loss_real + d_loss_fake) #+ d_loss_gp * 10
            
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            

            #################################### Train Generator ####################################
            #                                                                                       #
            #########################################################################################
            
            if i % 2 == 0:
                # create fake image
                x_fake = self.G(young_input)
                
                out_src = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
#                 g_loss_fake = self.MSE(out_src, old_label)
#                 g_loss_fake = torch.mean((out_src-1)**2)

        
                # reconstruction
                
                recons_input = torch.cat([x_fake, r_domain], dim = 1)
                x_reconstruction = self.G(recons_input)
                g_loss_rec = torch.mean(torch.abs(young_face - x_reconstruction))
                
                g_loss = g_loss_fake + g_loss_rec * 10

                    
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

            if i % 200 == 0:
                runtime = time.time() - start_time
                runtime = str(datetime.timedelta(seconds=runtime))[:-7]
                print('iter : {}, d_loss : {}, g_loss : {}, runtime : {}'.format(i, d_loss, g_loss, runtime))
                
                self.print_img(young_face)
                plt.show()
                self.print_img(x_fake)
                plt.show()
                self.print_img(x_reconstruction)
                plt.show()
                
                x_fake_list = [young_face]
                x_fake_list.append(x_fake)
                
                if i > 15000:
                    x_concat = torch.cat(x_fake_list, dim=3)
                    result_path = os.path.join('result_img4', '{}-images.jpg'.format(i//200))
                    save_image((x_concat.data.cpu()), result_path, nrow=1, padding = 0)
                    print('saved real and fake images into {}...'.format(result_path))
                
            
            
        def test(self):
            
            data_loader = self.data_loader
            
            with torch.no_grad():
                for i, (x_real,) in enumerate(data_loader):
                    
                    x_real = x_real.to(self.device)
                    
                    output = self.G(x_real)
                    
                    grays = torch.from_numpy(np.resize(output.numpy(), (self.batch_size, 1, 128, 128)))
                    out = torchvision.utils.make_grid(grays)
                    imshow(out)
                    plt.show()
                    


# In[ ]:


from torch.backends import cudnn
def main():
    cudnn.benchmark = True
    
    only_face_under25 = os.path.join('../../data/only_face','25')
    only_face_over60 = os.path.join('../../data/only_face','60')
    
    under25_128 = get_loader(img_dir = only_face_under25, label = 0, batch_size = 4)
    over60_128 = get_loader(img_dir = only_face_over60, label = 1, batch_size = 4)
    
    solver = Solver(young = under25_128, old = over60_128, batch_size = 4, g_lr = 0.0001, d_lr = 0.0001)
    
    solver.train()
    


# In[ ]:


if __name__ == '__main__':
    main()


# In[ ]:




