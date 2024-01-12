import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class LungOneDataset(Dataset):
    def __init__(self, fold=0, train=True):
        self.DATA_DIR  = "/Users/sudarshan/darshanz/datasets/lung1"
        label_df = pd.read_csv(f'{self.DATA_DIR}/label.csv')
        self.data_  = label_df[label_df['fold'] != fold] if train else label_df[label_df['fold'] == fold]
        self.event_status =  self.data['event_status']
        self.survival_time = self.data['survival_time'] 
        self.len = self.data['event_status'].shape[0] 
        
    def __getitem__(self, index):
        patient_id = self.data_.iloc[index]['patient_id']
        ct = torch.load(f'{self.DATA_DIR}/CT_ONLY/vols/{patient_id}.pt')
        return ct, self.data.iloc[index]['survival_time'], self.data.iloc[index]['event_status']
        
    def __len__(self):
        return self.len
    