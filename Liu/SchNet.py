import torch
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import time
from dig.threedgraph.dataset import Chem3D
from dig.threedgraph.method import SphereNet,SchNet,DimeNetPP,ComENet
from dig.threedgraph.method import run
from dig.threedgraph.evaluation import ThreeDEvaluator
import math
import os
def get_fold(size = 2479,fold=10):
    data = range(size)
    length = math.ceil((len(data) / fold))  # length of each fold
    folds = []
    for i in range(9):
        folds += [data[i * length:(i + 1) * length]]
    folds += [data[9 * length:len(data)]]
    return folds
def writetxt(best_trains,best_valids,best_tests,time,save_location):
    with open(save_location, 'w') as f:
        f.write(f'Trains: {str(best_trains)}, Valids: {str(best_valids)}, Tests: {str(best_tests)}, Time: {str(abs(time))}')


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    dataset = Chem3D(root='../Liu')
    dataset.data.y = dataset.data['y']
    best_tests = []
    best_valids = []
    best_trains = []
    modleName = 'SchNet'
    save_location = f'{os.getcwd()}/modelPerformance/2129Data/{modleName}/{modleName}.txt'

    folds = get_fold(size = len(dataset.data.y))
    split_idxs = dataset.get_idx_split(len(dataset.data.y),folds =folds)


    start = time.time()
    print(len(split_idxs),'folds')

    for curr_fold in range(len(split_idxs)):
        split_idx = split_idxs[curr_fold]
        train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        model_chem3d = SchNet(cutoff=5.0, num_layers=6, hidden_channels=256, num_filters=256, num_gaussians=60)
        loss_func = torch.nn.MSELoss()
        evaluation = ThreeDEvaluator()
        run3d = run()
        run3d.run(curr_fold,modleName,device, train_dataset, valid_dataset, test_dataset, model_chem3d, loss_func, evaluation, epochs=100,
                  batch_size=12, vt_batch_size=12, lr=0.00055, lr_decay_factor=0.7, lr_decay_step_size=10)
        best_tests.append(round(run3d.best_test,2))
        best_trains.append(round(run3d.best_train,2))
        best_valids.append(round(run3d.best_valid,2))

        curr = start-time.time()
        writetxt(best_trains, best_valids,best_tests, curr,save_location)
    print(best_trains)
    print(best_tests)
    end = time.time()
    print(start-end)

