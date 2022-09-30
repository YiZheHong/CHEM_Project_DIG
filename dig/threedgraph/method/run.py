import time
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import csv
import pandas as pd

class run():
    r"""
    The base script for running different 3DGN methods.
    """

    def __init__(self):
        pass
    def record(self):
        loc = f'{os.getcwd()}/modelPerformance/2129Data/{self.modelName}/{self.modelName}_fold{self.fold+1}'
        train_loc = loc+'_train'
        valid_loc = loc+'_valid'
        test_loc = loc+'_test'
        with open(train_loc+"Loss.txt", 'w') as f:
            f.write(str(self.train_loss))
        with open(valid_loc+"Loss.txt", 'w') as f:
            f.write(str(self.valid_loss))
        with open(test_loc+"Loss.txt", 'w') as f:
            f.write(str(self.test_loss))
        self.validDf.to_csv(valid_loc+"Result.csv",encoding='utf-8', index=False)
        self.testDf.to_csv(test_loc+"Result.csv",encoding='utf-8', index=False)

    def run(self, fold,modelName,device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=500,
            batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0.001,
            energy_and_force=False, p=100, save_dir='', log_dir=''):
        r"""
        The run script for training and validation.

        Args:
            device (torch.device): Device for computation.
            train_dataset: Training data.
            valid_dataset: Validation data.
            test_dataset: Test data.
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            loss_func (function): The used loss funtion for training.
            evaluation (function): The evaluation function. 
            epochs (int, optinal): Number of total training epochs. (default: :obj:`500`)
            batch_size (int, optinal): Number of samples in each minibatch in training. (default: :obj:`32`)
            vt_batch_size (int, optinal): Number of samples in each minibatch in validation/testing. (default: :obj:`32`)
            lr (float, optinal): Initial learning rate. (default: :obj:`0.0005`)
            lr_decay_factor (float, optinal): Learning rate decay factor. (default: :obj:`0.5`)
            lr_decay_step_size (int, optinal): epochs at which lr_initial <- lr_initial * lr_decay_factor. (default: :obj:`50`)
            weight_decay (float, optinal): weight decay factor at the regularization term. (default: :obj:`0`)
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            save_dir (str, optinal): The path to save trained models. If set to :obj:`''`, will not save the model. (default: :obj:`''`)
            log_dir (str, optinal): The path to save log files. If set to :obj:`''`, will not save the log files. (default: :obj:`''`)

        """
        self.fold = fold
        self.modelName = modelName
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.validDf = {}
        self.testDf = {}

        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        self.best_valid = float('inf')
        self.best_test = float('inf')


        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)

        for epoch in range(1, epochs + 1):
            print("\n=====Epoch {}".format(epoch), flush=True)

            print('\nTraining...', flush=True)
            train_mae = self.train(model, optimizer, train_loader, energy_and_force, p, loss_func, device)
            self.train_loss.append(train_mae)

            print('\n\nEvaluating...', flush=True)
            valid_mae,vDf = self.val(model, valid_loader, energy_and_force, p, evaluation, device)
            self.valid_loss.append(valid_mae)

            print('\n\nTesting...', flush=True)
            test_mae,tDf = self.val(model, test_loader, energy_and_force, p, evaluation, device)
            self.test_loss.append(test_mae)

            print()
            print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae})
            if log_dir != '':
                writer.add_scalar('train_mae', train_mae, epoch)
                writer.add_scalar('valid_mae', valid_mae, epoch)
                writer.add_scalar('test_mae', test_mae, epoch)

            if valid_mae < self.best_valid:
                self.best_valid = valid_mae
                self.best_test = test_mae
                self.best_train = train_mae
                self.validDf = vDf
                self.testDf = tDf
                if save_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': self.best_valid,
                                  'num_params': num_params}
                    torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))

            scheduler.step()
        self.record()
        print(f'Best validation RMSW so far: {self.best_valid}')
        print(f'Test RMSE when got best validation result: {self.best_test}')

        if log_dir != '':
            writer.close()

    def train(self, model, optimizer, train_loader, energy_and_force, p, loss_func, device):
        r"""
        The script for training.

        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in training.
            train_loader (Dataloader): Dataloader for training.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            loss_func (function): The used loss funtion for training. 
            device (torch.device): The device where the model is deployed.
        :rtype: Traning loss. ( :obj:`mae`)

        """
        model.train()
        loss_accum = 0
        for step, batch_data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            out = model(batch_data)
            MSE_loss = loss_func(out, batch_data.y.unsqueeze(1))
            RMSE_loss = torch.sqrt(MSE_loss)
            RMSE_loss.backward()
            optimizer.step()
            loss_accum += MSE_loss.detach().cpu().item()
        return np.sqrt(loss_accum / (step + 1))

    def val(self, model, data_loader, energy_and_force, p, evaluation, device):
        r"""
        The script for validation/test.

        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            data_loader (Dataloader): Dataloader for validation or test.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy. (default: :obj:`100`)
            evaluation (function): The used funtion for evaluation.
            device (torch.device, optional): The device where the model is deployed.
        :rtype: Evaluation result. ( :obj:`mae`)

        """
        model.eval()

        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)

        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            out = model(batch_data)
            preds = torch.cat([preds, out.detach_()], dim=0)
            targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)
        input_dict = {"y_true": targets, "y_pred": preds}

        trues = []
        preds = []
        for i in range(len(input_dict['y_true'])):
            trues.append(round(input_dict['y_true'][i].item(), 2))
            preds.append(round(input_dict['y_pred'][i].item(), 2))
        dic = {"y_true": trues, "y_pred": preds}
        df = pd.DataFrame.from_dict(dic)

        return evaluation.eval(input_dict)['rmse'],df