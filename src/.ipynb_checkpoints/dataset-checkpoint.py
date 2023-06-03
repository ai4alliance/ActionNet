import pandas as pd
from torchvision import transforms, models
from torch_snippets import fname
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

# AGU
def get_augment():
    return transforms.Compose([ 
        transforms.RandomOrder([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAutocontrast(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.3),
            transforms.RandomApply([transforms.Grayscale(3)], p=0.15),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            transforms.RandomPosterize(3, p=0.15),
            transforms.RandomAdjustSharpness(2.0, p=0.25),
            transforms.RandomEqualize(p=0.25),
        ]),
        transforms.RandomApply([transforms.RandomRotation(20, transforms.InterpolationMode.BICUBIC)], p=0.4),
        transforms.RandomPerspective(0.3, p=0.4),
    ])

# AUG1
# def get_augment():
#     return transforms.Compose([ 
#         transforms.RandomOrder([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomAutocontrast(p=0.5),
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
#             transforms.RandomApply([transforms.Grayscale(3)], p=0.15),
#             transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
#             transforms.RandomPosterize(4, p=0.15),
#             transforms.RandomAdjustSharpness(2.0, p=0.25),
#             transforms.RandomEqualize(p=0.25),
#         ]),
#         transforms.RandomApply([transforms.RandomRotation(20, transforms.InterpolationMode.BICUBIC)], p=0.4),
#         transforms.RandomPerspective(0.2, p=0.4),
#     ])
    

class HumanActionData(Dataset):
    def __init__(self, file_paths, df_path, cat2ind, augment, device):
        super().__init__()
        self.file_paths = file_paths
        self.cat2ind = cat2ind
        self.df = pd.read_csv(df_path)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize([224, 244]), 
            get_augment() if augment else transforms.RandomHorizontalFlip(),
            models.ResNet50_Weights.DEFAULT.transforms()
        ])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, ind):
        file_path = self.file_paths[ind]
        itarget = int(fname(file_path)[6:-4])
        target = self.df.iloc[itarget-1]['label']
        target = self.cat2ind[target]
        img = Image.open(file_path).convert('RGB')
        return img, target
    
    def collate_fn(self, data):
        imgs, targets = zip(*data)
        imgs = torch.stack([self.transform(img) for img in imgs], 0)
        imgs = imgs.to(self.device)
        targets = torch.tensor(targets).long().to(self.device)
        return imgs, targets
    
    def choose(self):
        return self[np.random.randint(len(self))]


# class HumanActionData(Dataset):
#     def __init__(self, file_paths, df_path, cat2ind, augment, device):
#         super().__init__()
#         self.file_paths = file_paths
#         self.cat2ind = cat2ind
#         self.df = pd.read_csv(df_path)
#         self.device = device
#         self.transform = transforms.Compose([
#             transforms.Resize([224, 244]), 
#             get_augment() if augment else transforms.RandomHorizontalFlip(),
#             models.ResNet50_Weights.DEFAULT.transforms()
#         ])
    
#     def __len__(self):
#         return len(self.file_paths)
    
#     def __getitem__(self, ind):
#         file_path = self.file_paths[ind]
#         itarget = int(fname(file_path)[6:-4])
#         target = self.df.iloc[itarget-1]['label']
#         target = self.cat2ind[target]
#         img = Image.open(file_path).convert('RGB')
#         return self.transform(img), torch.tensor(target).long()
    
#     def choose(self):
#         return self[np.random.randint(len(self))]


    

# from PIL import Image
# import matplotlib.pyplot as plt
# if __name__=='__main__':
    
#     img = Image.open('./Data/Human Action Recognition/test/Image_2842.jpg')
    
#     trans = get_augment()
#     for _ in range(5):
#         i = trans(img)
#         plt.imshow(i)
#         plt.tight_layout()
#         plt.axis('off')
#         plt.figure(figsize=(1,2))
#         plt.show()
#     pass
    



