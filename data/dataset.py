import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import random
import glob
import albumentations as A

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox, diagonal_corner, box_corner, empty_mask)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.tfs_mask = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            box_corner_mask = box_corner(self.image_size)
            diagonal_corner_mask = diagonal_corner(self.image_size)
            if np.random.randint(0,10)==0:
                mask = box_corner_mask
            elif np.random.randint(0,10)==1:
                mask = diagonal_corner_mask
            elif np.random.randint(0,10)==2:
                mask = empty_mask(self.image_size)
            else:
                mask = regular_mask | irregular_mask

        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        #Areas where mask=0, image remains. Areas where mask=1 noisy
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)
    
    
class ImageDataset(data.Dataset):
    def __init__(self, data_root, data_len=-1, image_size=[224, 224], transforms_mean=[0.485, 0.456, 0.406], transforms_std=[0.229, 0.224, 0.225], data_aug=True, loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=transforms_mean, std=transforms_std)
        ])
        self.aug = A.Compose([A.HorizontalFlip(p=0.5),A.VerticalFlip(p=0.5),A.RandomRotate90(p=0.5) ])
        self.loader = loader
        self.image_size = image_size
        self.data_aug = data_aug

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.loader(path)
        
        if self.data_aug:
            if isinstance(img, Image.Image):
                img = np.array(img)
            img = self.aug(image=img)['image']
            img = transforms.ToPILImage()(img)
        img = self.tfs(img)
        

        ret['img'] = img
        ret['path'] = path
        return ret

    def __len__(self):
        return len(self.imgs)
    
    
class AnomalyDataset(data.Dataset):
    def __init__(self, imgs_root, gts_root, data_len=-1, image_size=[224, 224], target_size=[256, 256], transforms_mean=[0.485, 0.456, 0.406], transforms_std=[0.229, 0.224, 0.225], loader=pil_loader):
        imgs = make_dataset(imgs_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
            
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=transforms_mean, std=transforms_std)
        ])
        self.tfs_mask = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((target_size[0], target_size[1])),
                transforms.ToTensor(),
        ])

        self.gts_root = gts_root
        self.loader = loader
        self.image_size = image_size
        self.target_size = target_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.loader(path)
        img = self.tfs(img)
        
        file_name = path.split('/')[-1]
        gt_path = os.path.join(self.gts_root, file_name)
        if os.path.exists(gt_path):
            gt = self.loader(gt_path)
            gt = self.tfs_mask(gt)
        else:
            gt = torch.zeros(1, self.target_size[0], self.target_size[1])          
        is_normal = torch.all(gt == 0)
        gt_label = (~is_normal).long()

        ret['img'] = img
        ret['img_path'] = path
        ret['gt'] = gt
        ret['gt_path'] = gt_path
        ret['gt_label'] = gt_label
        return ret

    def __len__(self):
        return len(self.imgs)