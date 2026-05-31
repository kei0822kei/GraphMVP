import argparse
import os
import pickle
import torch
import numpy as np
import h5py
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object.
    Preserves 1:1 mapping between graph nodes (atoms) and 3D positions.
    """
    # Atom features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # Bond features
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # Extract 3D positions
    conformer = mol.GetConformers()[0]
    positions = torch.tensor(conformer.GetPositions(), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, positions=positions)
    return data


class HDF5MoleculeDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        h5_path,
        n_mol,
        n_conf,
        n_upper,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.h5_path = h5_path
        self.n_mol = n_mol
        self.n_conf = n_conf
        self.n_upper = n_upper
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        data_list = []
        mol_counter = 0
        conf_counter = 0

        with h5py.File(self.h5_path, "r") as f:
            print(f"Processing HDF5 file: {self.h5_path}")
            for mol_id in tqdm(f.keys()):
                if mol_counter >= self.n_mol:
                    break

                grp = f[mol_id]
                mol = pickle.loads(bytes(grp["mol_bytes"][()]))

                conf_count = mol.GetNumConformers()
                if conf_count < self.n_conf or conf_count > self.n_upper:
                    continue

                # Process subset of conformers
                for i in range(min(conf_count, self.n_conf)):
                    conf_mol = Chem.Mol(mol)
                    conf_mol.RemoveAllConformers()
                    conf_mol.AddConformer(mol.GetConformer(i))

                    data = mol_to_graph_data_obj_simple_3D(conf_mol)
                    data.mol_id = mol_id
                    data_list.append(data)
                    conf_counter += 1

                mol_counter += 1

        print(f"Processed {mol_counter} molecules and {conf_counter} conformers.")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mol", type=int, help="number of unique molecules")
    parser.add_argument(
        "--n_conf", type=int, help="number of conformers of each molecule"
    )
    parser.add_argument("--n_upper", type=int, help="upper bound for conformers")
    parser.add_argument("--data_folder", type=str, help="folder containing dataset.h5")
    args = parser.parse_args()

    h5_path = os.path.join(args.data_folder, "dataset.h5")
    root_dir = os.path.join(
        args.data_folder,
        f"GEOM_H5_nmol{args.n_mol}_nconf{args.n_conf}_nupper{args.n_upper}",
    )

    dataset = HDF5MoleculeDataset(
        root=root_dir,
        h5_path=h5_path,
        n_mol=args.n_mol,
        n_conf=args.n_conf,
        n_upper=args.n_upper,
    )

    print(f"Dataset ready. Size: {len(dataset)}")
