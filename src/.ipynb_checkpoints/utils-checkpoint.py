import random, torch, os
import numpy as np
import time
import yaml
from addict import Dict
import pandas as pd
from datetime import datetime
import math
import json

def seed_torch(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        
def time_to_str(in_sec):
    return datetime.fromtimestamp(in_sec).strftime('%b %d %I:%M %p %y')


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


class ExpeLogger():
    def __init__(self, log_file, float_format='%0.8g', new_file=False):
        self.log_file = log_file
        self.float_format = float_format
        if new_file and os.path.exists(log_file): os.remove(log_file)
    
    def log(self, header, data):
        assert type(header) == dict
        if type(data) == dict:
            header.update(data)
            header['LogTime'] = time_to_str(time.time())
            newd = pd.DataFrame([header])
        else:
            assert type(data) == list
            l = []
            for i in data:
                h = header.copy()
                h.update(i)
                h['LogTime'] = time_to_str(time.time())
                l.append(h)
            newd = pd.DataFrame(l)
        
        if not os.path.exists(self.log_file):
            newd.to_csv(
                self.log_file, float_format=self.float_format, 
                index=True, index_label='ID')
            return
        df = pd.read_csv(self.log_file, index_col='ID')
        df = pd.concat([df, newd]).reset_index(drop=True)
        df.to_csv(
            self.log_file, float_format=self.float_format, 
            index=True, index_label='ID')
        return
    
    def exist(self, header):
        assert type(header) == dict
        headers = sorted(list(header))
        if not os.path.exists(self.log_file): return False
        log_df = pd.read_csv(self.log_file)
        if all(h in log_df.columns for h in headers):
            ids = log_df.apply(lambda x: '_'.join([str(x[h]) for h in headers]), axis=1)
            new_id = '_'.join([header[x] for x in headers])
            if new_id in ids.values: return True
        return False



def worker_init_fn(worker_id): 
    random.seed(worker_id)


class MyDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)


def read_yaml(fpath):
    with open(fpath, mode="r") as file:
        yml = yaml.safe_load(file)
        return MyDict(yml)
    
def write_yaml(fpath, data):
    with open(fpath, mode='w') as file:
        yaml.safe_dump(data, file, sort_keys = False)

def write_json(fpath:str, data:dict):
    with open(fpath, mode='w') as file:
        json.dump(data, file)


