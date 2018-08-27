import torch.utils.data as data
import numpy as np 
import os
import torchvision.transforms as transforms
from scipy import misc

def gaussian_noise(img):
    stddev = np.random.uniform(0,50)
    noise = np.random.randn(*img.shape) * stddev
    noise_img = np.clip(noise + img , 0 , 255).astype(np.uint8)
    return noise_img

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images


class NoiseDataLoader(data.Dataset):
    def __init__(self, opt , phase = 'train'):
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot , phase) # dataroot + /train or not (it is determined by opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.batch_size = opt.batch_size
        self.patch_num = opt.patch_num
        self.image_size = opt.image_size
        self.phase = phase


    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        img = misc.imread(AB_path) 
        img = img.reshape(img.shape[-1] , img.shape[0] , img.shape[1])  # img shape = [Color Channel , h , w ]
       
        h,w = img.shape[-2] , img.shape[-1]

        image_w = self.image_size
        image_h = self.image_size


        x = np.zeros((self.patch_num, 3, image_h , image_w), dtype=np.uint8)
        y = np.zeros((self.patch_num, 3, image_h , image_w), dtype=np.uint8)
        
        sample_id = 0 
                
        while True:
    
            if h >= image_h and w >= image_w:
                i = np.random.randint(h - image_h + 1)
                j = np.random.randint(w - image_w + 1)
                clean_patch = img[:, i:i + image_h, j:j + image_w]
                x[sample_id] = gaussian_noise(clean_patch) ## Noisy Input 
                y[sample_id] = clean_patch ## CLEAN TARGET
                sample_id += 1

                if sample_id == self.patch_num:
                    return AB_path , x, y  # X : Noisy Input / Y : Clean Target


    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'NoiseDataLoader'
