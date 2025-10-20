import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import rdkit.Chem as Chem

import torch
from torch_geometric.data import Data, InMemoryDataset


allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}


extended_periodic_table = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 
    110: "Ds", 111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
}
extended_periodic_table = {key - 1: value for key, value in extended_periodic_table.items()}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
    ]))

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
    ]))


class DARTDataset(InMemoryDataset):
    def __init__(self, root, tg_num, removeHs = False, transform = None, pre_transform = None, pre_fileter = None):
        '''
            dataset: tg414, tg421
        '''
        self.root = root
        self.tg = tg_num
        self.dataset = 'tg' + str(tg_num)
        self.removeHs = removeHs
        super(DARTDataset, self).__init__(root, transform, pre_transform, pre_fileter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, self.dataset, 'raw')
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dataset, 'processed')
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    @property
    def num_classes(self):
        return 2
    
    def _load_raw_dataset(self):
        path = f'../dataset/dart_{self.dataset}.xlsx'
        self.raw_data = pd.read_excel(path)
    
    def _atom_feature(self, mol):
        atom_feature_list = []
        for atom in mol.GetAtoms():
            atom_feature = [
                safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
                safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
                allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
            atom_feature_list.append(atom_feature)

        atom_feature_list = np.array(atom_feature_list)
        
        return atom_feature_list
    
    def smiles_to_graph(self, smiles_string):
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            raise ValueError('Invalid SMILES string')
        mol = mol if self.removeHs else Chem.AddHs(mol)
        
        x = self._atom_feature(mol)
        
        edge_index = []
        edge_feature_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append((i, j))
            edge_index.append((j, i))
            edge_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
            edge_feature_list.append(edge_feature)
            edge_feature_list.append(edge_feature)

        x = torch.tensor(x, dtype = torch.int32)
        edge_index = torch.tensor(edge_index, dtype = torch.long).t()
        edge_attr = torch.tensor(edge_feature_list, dtype = torch.long)
        
        data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr)
        
        return data

    def _construct_graphs(self):
        smiles = self.raw_data.SMILES
        
        data_list = []
        for i in range(len(self.raw_data)):
            data = self.smiles_to_graph(smiles[i])
            data.y = torch.tensor(self.raw_data.toxicity[i]).to(torch.long)
            data.smiles = smiles[i]
            data.idx = i
            
            data_list.append(data)
        
        return data_list
    
    def process(self):
        self._load_raw_dataset()
        data_list = self._construct_graphs()
        
        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
        
        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == '__main__':
    root = '../../dataset'
    
    dataset = DARTDataset('../dataset', 414)
    print(dataset[0])
