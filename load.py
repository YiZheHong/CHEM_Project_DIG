import os
import os.path as osp
import torch
from sklearn.utils import shuffle
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data


class Chem3D(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`QM9` dataset
        which is from `"Quantum chemistry structures and properties of 134 kilo molecules" <https://www.nature.com/articles/sdata201422>`_ paper.
        It connsists of about 130,000 equilibrium molecules with 12 regression targets:
        :obj:`mu`, :obj:`alpha`, :obj:`homo`, :obj:`lumo`, :obj:`gap`, :obj:`r2`, :obj:`zpve`, :obj:`U0`, :obj:`U`, :obj:`H`, :obj:`G`, :obj:`Cv`.
        Each molecule includes complete spatial information for the single low energy conformation of the atoms in the molecule.

        .. note::
            We used the processed data in `DimeNet <https://github.com/klicperajo/dimenet/tree/master/data>`_, wihch includes spatial information and type for each atom.
            You can also use `QM9 in Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9>`_.


        Args:
            root (string): the dataset folder will be located at root/qm9.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        Example:
        --------

        Batch(Cv=[32], G=[32], H=[32], U=[32], U0=[32], alpha=[32], batch=[579], gap=[32], homo=[32], lumo=[32], mu=[32], pos=[579, 3], ptr=[33], r2=[32], y=[32], z=[579], zpve=[32])

        Where the attributes of the output data indicates:

        * :obj:`z`: The atom type.
        * :obj:`pos`: The 3D position for atoms.
        * :obj:`y`: The target property for the graph (molecule).
        * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs
    """



    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # return 'CHEM_Data2.csv'
        return 'CHEM_Data_2129.csv'

    @property
    def processed_file_names(self):
        return '3d_chem_graphs.pt'


    def process(self):
        path = osp.join(self.raw_dir, self.raw_file_names)
        print(path)
        data = pd.read_csv(path)
        data_list = []
        print(len(data))
        for graph_index in range(0, len(data)):
            pre_name = 'Chemicals_'
            R = []
            count = 0
            for chem_num in range(1, 8):
                chem = pre_name + str(chem_num)
                if pd.notnull(data.iloc[graph_index][chem]):
                    count += 1
                    str_list = data.iloc[graph_index][chem]
                    atoms = str_list[1:len(str_list) - 1].split(',')
                    for pos in range(1, 4):
                        atoms[pos] = float(atoms[pos][2:len(atoms[pos]) - 1])
                    R.append(atoms[1:])

            R = torch.tensor(R, dtype=torch.float32)
            z_i = torch.tensor(count * [1], dtype=torch.int64)
            y_i = torch.tensor(data['Y_value'].iloc[graph_index], dtype=torch.float32)
            v_i = torch.tensor(data['Vibration mode'].iloc[graph_index], dtype=torch.float32)
            d_i = torch.tensor(data['Displacement'].iloc[graph_index], dtype=torch.float32)



            # single_data = Data(pos=R, z=z_i, y=y_i,v=v_i,d=d_i,m=m_i)
            single_data = Data(pos=R, z=z_i, y=y_i,v=v_i,d=d_i)

            data_list.append(single_data)
        data, slices = self.collate(data_list)
        print(data)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        print(ids)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict


if __name__ == '__main__':
    print(os.getcwd())
    dataset = Chem3D('Liu')
    print(dataset.process())
    # print(dataset.data.v)
    # print(dataset.data.pos.shape)
    print(dataset.data.y)
    # split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1434, valid_size=307, seed=42)
    # print(dataset[split_idx['test']])
    # train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    #
    # # train_loader = loader.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # # data = next(iter(train_loader))
    # # print(data)
    # model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
    #                   hidden_channels=128, out_channels=1, int_emb_size=64,
    #                   basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
    #                   num_spherical=3, num_radial=6, envelope_exponent=5,
    #                   num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True
    #                   )
    # loss_func = torch.nn.L1Loss()
    # evaluation = ThreeDEvaluator()
