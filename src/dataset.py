import pandas as pd
from torchvision import transforms, models
from torch_snippets import fname
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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


class Pass(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def __call__(self, x):
        return x
    

class HumanActionData(Dataset):
    def __init__(self, file_paths, df_path, cat2ind, augment, device):
        super().__init__()
        self.file_paths = file_paths
        self.cat2ind = cat2ind
        self.df = pd.read_csv(df_path)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize([224, 244]), 
            get_augment() if augment else Pass(),
            models.ResNet50_Weights.DEFAULT.transforms()
        ])
        
        self.data, self.label = [], []
        for ind in tqdm(range(len(self.file_paths))):
            file_path = self.file_paths[ind]
            
            #print(file_path)
            itarget = int(fname(file_path)[6:-4])
            target = self.df.iloc[itarget-1]['label']
            target = self.cat2ind[target]
            img = Image.open(file_path).convert('RGB')
            
            self.data.append(img)
            self.label.append(target)
            
            
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, ind):
        img = self.data[ind]
        img = self.transform(img)
        
        target = self.label[ind]
        target = torch.tensor(target).long()
        return img, target
    

    def choose(self):
        return self[np.random.randint(len(self))]


    

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
    



