import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from utils import read_config, create_logger 
from experiments.basic_3d import Basic3DModelRunner


import sys


seed = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
     
    configs_dir = 'configs'
    params = {
              'A':f'{configs_dir}/expt_a_config.ini'
              }
    
    

    args = sys.argv[1:]
    
    if len(args) == 0:
        print("Please provide name of the method: eg. main.py A")
    elif args[0] in list(params.keys()):
        print("PyTorch Version: ",torch.__version__)
        seed_everything(42)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'mps'
        logs_dir='logs'
        logger = create_logger(logs_dir)
        logger.info('Running {} ({})...'.format(args[0], params[args[0]]))
        if args[0] in ['A']:
            basic3d = Basic3DModelRunner(device, params[args[0]])             
            basic3d.train()
            
    else:
        print(f"Unknown method: '{args[0]}',  Supported methods: {list(params.keys())} ")