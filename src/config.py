import os


class CFG:
    group = 'ResNET'    # Exp name
    name = 'base'      # Sub exp name
    
    
    DIR = "./Data/Human Action Recognition/"
    TRAIN_DIR=f"{DIR}train"
    TEST_DIR=f"{DIR}test"
    TRAIN_VAL_DF = "./Data/Human Action Recognition/Training_set.csv"
        
    
    batch_size = 2
    learning_rate = 0.001
    dropout = 0.2
    weight_decay = 1e-5
    hidden_size = 256
    
    num_workers = len(os.sched_getaffinity(0))

