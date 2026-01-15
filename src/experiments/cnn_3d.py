import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
import torch
from torch import nn 
from torch.utils.data import DataLoader
from lung_dataset import LungOneDataset 
from models import CNN3D
from utils import read_config 
import torch.optim as optim
from sklearn.metrics import f1_score
import wandb
import copy
import time
from tqdm import  tqdm
from loss import CensoredCrossEntropyLoss
from lifelines.utils import concordance_index

class Basic3DModelRunner():
    '''
    Baseline Trainer
    '''
    def __init__(self, device, conf_file):
        super(Basic3DModelRunner, self).__init__()
        wandb.init(project="lung_basic3d")
        config = read_config(conf_file)
        self.device = device
        print(self.device)
        self.conf = config['train']
        self.model_path = f'logs/saved_model/saved_model_{int(time.time())}.pt'


    def train(self):
        # Data Loaders
        train_dataset = LungOneDataset(fold=0)
        val_dataset = LungOneDataset(fold= 0, train=False) 
        trainloader = DataLoader(train_dataset, batch_size=4)
        testloader = DataLoader(val_dataset, batch_size=4) 
        dataloaders = {"train": trainloader, "val": testloader}

         
        model = CNN3D().to(self.device)
        optimizer = optim.Adam(model.parameters(), self.conf['learning_rate'])

        criterion = CensoredCrossEntropyLoss()
        NUM_EPOCHS = self.conf['epochs']
        best_ci = 0.0
 
        best_model_wts = copy.deepcopy(model.state_dict())
        epoch_tqdm = tqdm(total=NUM_EPOCHS, desc='Epoch', position=0)
        training_info = tqdm(total=0, position=1, bar_format='{desc}')
        for epoch in range(NUM_EPOCHS):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()   
                else:
                    model.eval()   

                running_loss = 0.0

                gt_train = []
                pred_train = []
                gt_val = []
                pred_val = []
                events_train = []
                events_val = []

                for ct, y, e in dataloaders[phase]:  
                    ct = ct.to(self.device)  
                    e = e.to(self.device) #event
                    labels = y.to(self.device)
 
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(ct)
                        loss = criterion(outputs, labels, e)

                        if phase == 'val':
                            gt_val.extend(labels.detach().cpu())
                            events_val.extend(e.detach().cpu())
                            pred_val.extend(np.argmax(outputs.detach().cpu(), axis=1))
                        else:
                            gt_train.extend(labels.detach().cpu())
                            events_train.extend(e.detach().cpu())
                            pred_train.extend(np.argmax(outputs.detach().cpu(), axis=1))

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * y.size(0)

                total_count = len(train_dataset) if phase == 'train' else len(val_dataset)
                epoch_loss = running_loss / total_count

                if phase == 'train':
                    train_ci = concordance_index(gt_train, pd.DataFrame(pred_train.detach().numpy()), events_train)
                     

                if phase == 'val': 
                    val_ci = concordance_index(gt_val, pd.DataFrame(pred_val.detach().numpy()), events_val)

                    if val_ci > best_ci:
                        best_ci = val_ci
                        best_model_wts = copy.deepcopy(model.state_dict())

            training_info.set_description_str(
                f'Epoch {epoch + 1}/{NUM_EPOCHS},  Loss:{epoch_loss:.4f}, C-index(Train):{train_ci:.4f},  C-index(Val):{val_ci:.4f}')
            wandb.log({"loss":epoch_loss, "train_f1": train_ci, "val_f1": val_ci})
            epoch_tqdm.update(1)

            # load best model weights and save
        model.load_state_dict(best_model_wts)
        torch.save(model, f'../logs/saved_model/saved_model_base3d.pt')

        self.plot_test_results(model)

    def plot_test_results(self, model): 
        val_dataset = LungOneDataset(fold= 0, train=False)
        testloader = DataLoader(val_dataset, batch_size=4)
        model.eval()  

        label_list = []
        pred_labels = []
        events_test = []

        for ct, y, e in testloader:
            ct = ct.to(self.device) 
            e = e.to(self.device) #event
            labels = y.to(self.device)
            with torch.set_grad_enabled(False):
                outputs = model(ct)

                label_list.extend(labels.detach().cpu())
                preds = np.argmax(outputs.detach().cpu(), axis=1)
                events_test.extend(e.detach().cpu())
                pred_labels.extend(preds)

        test_ci = concordance_index(label_list, pd.DataFrame(pred_labels.detach().numpy()), events_test)

        wandb.log({"test_ci": test_ci})
        print("test_ci:", test_ci)