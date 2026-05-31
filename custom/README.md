# Pre-Training with Custom Dataset

## Custom Dataset Preparation

To use a custom dataset with **GraphMVP**, data must be stored in HDF5 (`.h5`) format. Each molecule should be represented as a group within the HDF5 file, keyed by a unique identifier.

### Data Structure Requirements

Each HDF5 group must contain the following datasets and attributes:

| Key | Type | Description |
| --- | --- | --- |
| `id` (Group Name) | String | A unique identifier (e.g., a SHA-256 hash of the SMILES string). |
| `smiles` (Attribute) | String | The canonical SMILES string. |
| `numbers` (Dataset) | `int32` | Array of atomic numbers representing the molecule's graph. |
| `positions` (Dataset) | `float32` | Tensor of shape `(N_conformers, N_atoms, 3)` containing atomic coordinates. |

*Note: You may include additional attributes or datasets within the group as needed for your specific task.*

### Python Implementation Example

The following script demonstrates how to process a list of molecular data (already containing SMILES, atomic numbers, and a list of `(N_atoms, 3)` position arrays) into the required HDF5 format:

```python
import numpy as np
import h5py
import hashlib

def hash_smiles(smiles):
    return hashlib.sha256(smiles.encode()).hexdigest()

def save_to_h5(molecule_data_list, output_filename="dataset.h5"):
    """
    Args:
        molecule_data_list: A list of dicts, each containing:
            - "smiles": str
            - "numbers": np.array (shape: [N_atoms], dtype: int32)
            - "positions_list": list of np.array (each shape: [N_atoms, 3], dtype: float32)
    """
    with h5py.File(output_filename, "w") as f:
        for data in molecule_data_list:
            smiles = data["smiles"]
            numbers = data["numbers"]
            # Convert list of position arrays to a single numpy tensor
            positions = np.array(data["positions_list"], dtype=np.float32)
            
            # Generate ID and create group
            id_ = hash_smiles(smiles)
            grp = f.create_group(id_)
            
            # Save attributes and datasets
            grp.attrs["smiles"] = smiles
            grp.create_dataset("numbers", data=numbers)
            grp.create_dataset("positions", 
                               data=positions, 
                               compression="gzip")

# Example usage:
# save_to_h5(your_data_list, "sampledb.h5")

```

* **Compression:** We strongly recommend using `compression="gzip"` for the `positions` dataset to significantly reduce file size without losing precision.
* **Consistency:** Ensure that the atom ordering in the `numbers` array matches the ordering in the `positions` coordinates for every conformer.
* **Uniqueness:** Ensure that the `id` used for the HDF5 group name is globally unique within your dataset to prevent data overwriting.

After preparation, dataset for GraphMVP is build with the following:

```python
# python GEOM_dataset_preparation.py --n_mol 50000 --n_conf 5 --n_upper 1000 --data_folder hoge
python custom_dataset_preparation.py --n_mol 50000 --n_conf 5 --n_upper 1000 --data_folder hoge
```

## Run

TODO: Future edit.

```python
python pretrain_GraphMVP_hybrid.py --epochs=100 --dataset=GEOM_3D_nmol50000_nconf5_nupper1000_morefeat --batch_size=256 --SSL_masking_ratio=0.15 --CL_similarity_metric=EBM_dot_prod --T=0.1 --normalize --AE_model=VAE --AE_loss=l2 --detach_target --beta=1 --alpha_1=1 --alpha_2=1 --SSL_2D_mode=CP --alpha_3=1 --num_interactions=6 --num_gaussians=51 --cutoff=10 --schnet_lr_scale=0.1 --dropout_ratio=0 --num_workers=8 --output_model_dir=../output/GraphMVP_Hybrid_C/pretraining
```
