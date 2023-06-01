
import pandas as pd
import os, gc, argparse
from config import CFG
import torch
import torch
from torch import nn, optim
from torchvision import transforms, models
from torch_snippets import Report
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary 

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import cv2
from glob import glob
import pandas as pd

from model import ActionClassifier
from dataset import HumanActionData
import wandb

sns.set_theme()
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def make_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--name', type=str, help='Name of the experiment; (default: %(default)s)', required=True)
    arg('--num_workers', type=int, default=CFG.num_workers, help='Number of dataloader workers; (default: %(default)s)')
    arg('--data_root', type=str, default=CFG.data_root, help='Location where data is stored; (default: %(default)s)')
    arg('--log_file', type=str, default=CFG.log_file, help='On which file(.csv) to store logs; (default: %(default)s)')
    arg('--output_dir', type=str, default='./outputs/models', help='Where to save outputs; (default: %(default)s)')
    arg("--save_best", action="store_true", help="Save best model weights")
    arg("--save_latest", action="store_true", help="Save latest model weights")
    arg("--force_run", action="store_true", help="Don not skip this experiment")
    arg("--debug", action="store_true", help="debug")
    arg("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Which device to train on; (default: %(default)s)")
    return parser




def train(data, classifier, optimizer, loss_fn):
    classifier.train()
    imgs, targets = data
    outputs = classifier(imgs)
    loss = loss_fn(outputs, targets)
    preds = outputs.argmax(-1)
    acc = (sum(preds==targets) / len(targets))
    classifier.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc


@torch.no_grad()
def validate(data, classifier, loss_fn):
    classifier.eval()
    imgs, targets = data
    outputs = classifier(imgs)
    loss = loss_fn(outputs, targets)
    preds = outputs.argmax(-1)
    acc = (sum(preds==targets) / len(targets))
    return loss, acc






if __name__ == '__main__':
    
    train_val_data=glob(CFG.TRAIN_DIR+'/*.jpg')
    train_data, val_data = train_test_split(train_val_data, test_size=0.15, shuffle=True)
    print('Train Size', len(train_data))
    print('Val Size', len(val_data))
    
    df=pd.read_csv(f"{CFG.DIR}Training_set.csv")
    agg_labels = df.groupby('label').agg({'label': 'count'})
    agg_labels.rename(columns={'label': 'count'})
    
    
    ind2cat = sorted(df['label'].unique().tolist())
    cat2ind = {cat: ind for ind, cat in enumerate(ind2cat)}
    
    
    # Debug
    train_data = train_data[:4]
    val_data = val_data[:4]
    
    
    train_ds = HumanActionData(train_data, CFG.TRAIN_VAL_DF, cat2ind)
    train_dl = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        collate_fn=train_ds.collate_fn,
        drop_last=True, 
    )
    
    val_ds = HumanActionData(val_data, CFG.TRAIN_VAL_DF, cat2ind)
    val_dl = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False,
        collate_fn=val_ds.collate_fn,
        drop_last=False
    )
    
    
    n_epochs = 2
    log = Report(n_epochs)
    classifier = ActionClassifier(len(ind2cat)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    #log.wandb_logging=True
    
    for epoch in range(n_epochs):
        n_batch = len(train_dl)
        for i, data in enumerate(train_dl):
            train_loss, train_acc = train(data, classifier, optimizer, loss_fn)
            pos = epoch + ((i+1)/n_batch)
            log.record(pos=pos, train_loss=train_loss, train_acc=train_acc, end='\r')
            
        n_batch = len(val_dl)
        for i, data in enumerate(val_dl):
            val_loss, val_acc = validate(data, classifier, loss_fn)
            pos = epoch + ((i+1)/n_batch)
            log.record(pos=pos, val_loss=val_loss, val_acc=val_acc, end='\r')
        
        scheduler.step()
        log.report_avgs(epoch+1)
    
    
    
    if not os.path.exists('./Output'): os.path.mkdir('./Output')
    torch.save(classifier.state_dict(), './Output/model_weights.pth')
    
    
    
    pass
    
    



