from config import CFG
from torch import nn, optim
from torch_snippets import Report
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
from model import ActionClassifier
from dataset import HumanActionData
from utils import AverageMeter
import pandas as pd
import os, argparse
import torch, wandb


def make_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--group', type=str, default=CFG.group, help='Group of the experiment; (default: %(default)s)')
    arg('--name', type=str, default=CFG.name, help='Name of the experiment; (default: %(default)s)')
    # hyper parameters
    arg('--n_epochs', type=str, default=CFG.n_epochs, help='Number of Epoch; (default: %(default)s)')
    arg('--batch_size', type=str, default=CFG.batch_size, help='Batch Size; (default: %(default)s)')
    arg('--learning_rate', type=str, default=CFG.learning_rate, help='Learning Rate; (default: %(default)s)')
    arg('--dropout', type=str, default=CFG.dropout, help='Dropout; (default: %(default)s)')
    arg('--weight_decay', type=str, default=CFG.weight_decay, help='Weight Decay; (default: %(default)s)')
    arg('--hidden_size', type=str, default=CFG.hidden_size, help='Hidden Size; (default: %(default)s)')
    arg("--augment", action="store_true", help="debug")
    # others
    arg('--num_workers', type=int, default=CFG.num_workers, help='Number of dataloader workers; (default: %(default)s)')
    arg("--debug", action="store_true", help="debug")
    arg("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Which device to train on; (default: %(default)s)")
    return parser


def train(data, model, optimizer, loss_fn, scaler):
    imgs, targets = data
    
    model.train()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=CFG.amp):
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    preds = outputs.argmax(-1)
    acc = (sum(preds==targets) / len(targets))
    return loss, acc


@torch.no_grad()
def validate(data, classifier, loss_fn):
    classifier.eval()
    imgs, targets = data
    outputs = classifier(imgs)
    loss = loss_fn(outputs, targets)
    preds = outputs.argmax(-1)
    acc = (sum(preds==targets) / len(targets))
    return loss, acc, outputs.to('cpu').numpy(), targets.to('cpu').numpy()



if __name__ == '__main__':
    args = make_parser().parse_args()
    device = args.device
    
    
    wandb.login()
    run = wandb.init(
        project='ActionNet',
        name=args.name,
        config = {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'hidden_size': args.hidden_size,
            'dropout': args.dropout,
            'weight_decay': args.weight_decay,
            
            'test_size': CFG.test_size,
            'valid_size': CFG.valid_size,
            'test_split_seed': CFG.test_split_seed,
            'valid_split_seed': CFG.valid_split_seed,
        },
        group=args.group,
        anonymous=None
    )
    
    
    train_test_data = glob(CFG.TRAIN_DIR+'/*.jpg')
    train_val_data, test_data = train_test_split(
        train_test_data, 
        test_size=CFG.test_size, 
        shuffle=True, 
        random_state=CFG.test_split_seed
    )
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=CFG.valid_size, 
        shuffle=True, 
        random_state=CFG.valid_split_seed
    )
    
    
    print('Train Size', len(train_data))
    print('Val Size', len(val_data))
    
    df=pd.read_csv(CFG.TRAIN_VAL_DF)
    agg_labels = df.groupby('label').agg({'label': 'count'})
    agg_labels.rename(columns={'label': 'count'})
    
    
    ind2cat = sorted(df['label'].unique().tolist())
    cat2ind = {cat: ind for ind, cat in enumerate(ind2cat)}
    
    
    # Debug
    if args.debug:
        train_data = train_data[:16]
        val_data = val_data[:8]
        test_data = test_data[:8]
    
    
    train_ds = HumanActionData(train_data, CFG.TRAIN_VAL_DF, cat2ind, args.augment)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_ds.collate_fn,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    val_ds = HumanActionData(val_data, CFG.TRAIN_VAL_DF, cat2ind, False)
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=val_ds.collate_fn,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    test_ds = HumanActionData(test_data, CFG.TRAIN_VAL_DF, cat2ind, False)
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=val_ds.collate_fn,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    
    log = Report(args.n_epochs)
    model = ActionClassifier(
        hidden_size=args.hidden_size, 
        dropout=args.dropout, 
        ntargets=len(ind2cat)
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.amp)
    
    for epoch in range(args.n_epochs):
        
        # Train Model
        train_losses, train_accs = AverageMeter(), AverageMeter()
        n_batch = len(train_dl)
        for i, data in enumerate(train_dl):
            train_loss, train_acc = train(data, model, optimizer, loss_fn, scaler)
            train_losses.update(train_loss.item(), n_batch)
            train_accs.update(train_acc, n_batch)
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc})            
        
        # Validate Model
        val_losses, valid_accs = AverageMeter(), AverageMeter()
        n_batch = len(val_dl)
        for i, data in enumerate(val_dl):
            val_loss, val_acc, pred, target = validate(data, model, loss_fn)
            val_losses.update(val_loss.item(), n_batch)
            valid_accs.update(val_acc, n_batch)
            wandb.log({"epoch": epoch+1, "valid_loss": val_loss, "val_acc": val_acc})
        
        # Log Results
        print(f'EPOCH: {epoch: >2}  train_loss:{train_losses.avg: .3f}  valid_loss:{val_losses.avg: .3f}  train_acc:{train_accs.avg: .4f}  valid_accs:{valid_accs.avg: .4f}')
        wandb.log({
            "epoch": epoch+1,
            "train_loss(avg)": train_losses.avg,
            "valid_loss(avg)": val_losses.avg,
            "train_acc(avg)": train_accs.avg,
            "valid_acc(avg)": valid_accs.avg,
            "learning_rate": scheduler.get_last_lr(),
        })
        
        scheduler.step()
    
    
    # Calculate Test Accuracy
    test_accs = AverageMeter()
    n_batch = len(test_dl)
    for i, data in enumerate(test_dl):
        _, test_acc, pred, target = validate(data, model, loss_fn)
        test_accs.update(test_acc, n_batch)
    wandb.log({"test_acc": test_accs.avg})
    
    
    if not os.path.exists('./Output'): os.path.mkdir('./Output')
    torch.save(model.state_dict(), './Output/model_weights.pth')
    wandb.finish()

