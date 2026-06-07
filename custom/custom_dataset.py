import argparse
import os
import torch
import numpy as np
import h5py
import logging
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


def mol_to_graph_data_obj_simple_3d(mol, positions):
    """
    Converts rdkit mol object to graph Data object.
    """
    # Atom features
    atom_features_list = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # Bond features
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = bond_to_feature_vector(bond)
            edges_list.extend([(i, j), (j, i)])
            edge_features_list.extend([feat, feat])
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        positions=torch.tensor(positions, dtype=torch.float),
    )


class CustomDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        n_mol,
        n_conf,
        n_upper,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.n_mol = n_mol
        self.n_conf = n_conf
        self.n_upper = n_upper
        # Define unique subdirectory for these parameters
        self.subdir = f"nmol{n_mol}_nconf{n_conf}_nupper{n_upper}"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["dataset.h5"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_dir(self):
        # Create a unique directory per parameter set
        return os.path.join(self.root, "processed", self.subdir)

    def process(self):
        data_list = []
        mol_counter = 0
        conf_counter = 0
        h5_path = os.path.join(self.raw_dir, "dataset.h5")

        logger.info(f"Opening HDF5 file: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            for mol_id in tqdm(f.keys(), desc="Processing molecules"):
                if mol_counter >= self.n_mol:
                    break

                grp = f[mol_id]
                smiles = grp.attrs["smiles"]

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                positions_all = grp["positions"][:]
                energies_all = grp["energies"][:]
                conf_count = positions_all.shape[0]

                if conf_count < self.n_conf or conf_count > self.n_upper:
                    continue

                sorted_indices = np.argsort(energies_all)
                selected_indices = sorted_indices[: self.n_conf]

                for i in selected_indices:
                    data = mol_to_graph_data_obj_simple_3d(mol, positions_all[i])
                    data.mol_id = mol_id
                    data.energy = float(energies_all[i])
                    data_list.append(data)
                    conf_counter += 1

                mol_counter += 1

        logger.info(f"Processed {mol_counter} molecules and {conf_counter} conformers.")

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"Saved to {self.processed_paths[0]}")


if __name__ == "__main__":
    # Configure logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_mol", type=int, required=True)
    parser.add_argument("--n_conf", type=int, required=True)
    parser.add_argument("--n_upper", type=int, required=True)
    parser.add_argument(
        "--data_folder", type=str, required=True, help="Path to 'dataset/sample'"
    )
    args = parser.parse_args()

    # Initialization automatically checks processed/nmolXX.../data.pt
    dataset = CustomDataset(
        root=args.data_folder,
        n_mol=args.n_mol,
        n_conf=args.n_conf,
        n_upper=args.n_upper,
    )

    logger.info(f"Dataset ready at {dataset.processed_dir}. Size: {len(dataset)}")
