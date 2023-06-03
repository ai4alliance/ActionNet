import os


class CFG:
    group = 'ResNET'    # Exp name
    name = 'base'      # Sub exp name
    amp = True
    
    DIR = "./Data/Human Action Recognition"
    TRAIN_DIR=f"{DIR}/train"
    TEST_DIR=f"{DIR}/test"
    TRAIN_VAL_DF = f"{DIR}/labels.csv"
    
    test_size = 0.15
    valid_size = 0.15
    test_split_seed = 42
    valid_split_seed = 42
    
    n_epochs = 6
    batch_size = 2
    learning_rate = 0.001
    train_last_nlayer = 0
    dropout = 0.2
    weight_decay = 1e-5
    hidden_size = 256
    
    num_workers = len(os.sched_getaffinity(0))

