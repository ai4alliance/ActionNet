
import pandas as pd
import os, gc, argparse
from config import CFG
import torch
import torch
from torch import nn, optim
from torchvision import transforms, models
from torch_snippets import Report
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from glob import glob

from model import ActionClassifier
from dataset import HumanActionData
import wandb

def make_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--name', type=str, default=CFG.name, help='Name of the experiment; (default: %(default)s)')
    arg('--num_workers', type=int, default=CFG.num_workers, help='Number of dataloader workers; (default: %(default)s)')
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
    args = make_parser().parse_args()
    device = args.device
    
    wandb.login()
    run = wandb.init(
        project='ActionNet',
        name=CFG.name,
        config = {
            'learning_rate': CFG.learning_rate,
            'batch_size': CFG.batch_size,
            'hidden_size': CFG.hidden_size,
            'dropout': CFG.dropout,
            'weight_decay': CFG.weight_decay,
        },
        group=CFG.group,
        anonymous=None
    )
    
    
    
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
    if args.debug:
        train_data = train_data[:4]
        val_data = val_data[:4]
    
    
    train_ds = HumanActionData(train_data, CFG.TRAIN_VAL_DF, cat2ind)
    train_dl = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        collate_fn=train_ds.collate_fn,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    val_ds = HumanActionData(val_data, CFG.TRAIN_VAL_DF, cat2ind)
    val_dl = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False,
        collate_fn=val_ds.collate_fn,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    
    n_epochs = 2
    log = Report(n_epochs)
    model = ActionClassifier(CFG.hidden_size, len(ind2cat)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    

    for epoch in range(n_epochs):
        n_batch = len(train_dl)
        for i, data in enumerate(train_dl):
            train_loss, train_acc = train(data, model, optimizer, loss_fn)
            pos = epoch + ((i+1)/n_batch)
            log.record(pos=pos, train_loss=train_loss, train_acc=train_acc, end='\r')
            wandb.log({"Epoch": epoch, "Train Loss": train_loss, "Train Accuracy": train_acc})            
            
        n_batch = len(val_dl)
        for i, data in enumerate(val_dl):
            val_loss, val_acc = validate(data, model, loss_fn)
            pos = epoch + ((i+1)/n_batch)
            log.record(pos=pos, val_loss=val_loss, val_acc=val_acc, end='\r')
            wandb.log({"Epoch": epoch, "Valid Loss": val_loss, "Valid Accuracy": val_acc})
        
        scheduler.step()
        log.report_avgs(epoch+1)
    
    

    if not os.path.exists('./Output'): os.path.mkdir('./Output')
    torch.save(model.state_dict(), './Output/model_weights.pth')
    wandb.finish()

