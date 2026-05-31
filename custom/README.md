# Pre-Training with Custom Dataset

## Custom Dataset Preparation

To use a custom dataset with **GraphMVP**, data must be stored in HDF5 (`.h5`) format. Each molecule is stored as a serialized RDKit `Mol` object, which preserves the strict 1:1 mapping between graph topology and spatial coordinates.

### Data Structure Requirements

| Key | Type | Description |
| --- | --- | --- |
| `id` (Group Name) | `string` | A unique identifier (a SHA-256 hash of the **canonical** SMILES string). |
| `smiles` (Attribute) | `string` | The canonical SMILES string. |
| `mol_bytes` (Dataset) | `blob` | Serialized RDKit `Mol` object (containing all conformers). |

### Python Implementation Example

The following script converts your molecular data into a GEOM-compatible HDF5 format, ensuring that all SMILES strings and hashes are derived from their canonical representation.

```python
import numpy as np
import h5py
import hashlib
import pickle
from rdkit import Chem

def hash_smiles(smiles):
    return hashlib.sha256(smiles.encode()).hexdigest()

def get_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

def save_to_h5(molecule_data_list, output_filename="dataset.h5"):
    """
    Args:
        molecule_data_list: A list of dicts, each containing:
            - "smiles": str (input SMILES)
            - "numbers": np.array (shape: [N_atoms], dtype: int32)
            - "positions": np.array (shape: [N_conformers, N_atoms, 3], dtype: float32)
    """
    with h5py.File(output_filename, "w") as f:
        for data in molecule_data_list:
            # Generate Canonical SMILES
            canonical_smiles = get_canonical_smiles(data["smiles"])
            if canonical_smiles is None: continue
            
            numbers = data["numbers"]
            positions = data["positions"]

            # 1. Create RDKit molecule from Canonical SMILES
            mol = Chem.MolFromSmiles(canonical_smiles)
            mol = Chem.AddHs(mol)
            
            # 2. Attach conformers
            for pos in positions:
                conf = Chem.Conformer(mol.GetNumAtoms())
                for i in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, pos[i])
                mol.AddConformer(conf, assignId=True)
            
            # 3. Serialize the RDKit Mol object
            mol_bytes = pickle.dumps(mol)

            # 4. Save to HDF5 using hash of Canonical SMILES
            id_ = hash_smiles(canonical_smiles)
            if id_ in f: continue
            
            grp = f.create_group(id_)
            grp.attrs["smiles"] = canonical_smiles
            grp.create_dataset("mol_bytes", data=np.void(mol_bytes))

# Example usage:
# save_to_h5(your_data_list, "sampledb.h5")

```

### Important Note on Data Integrity

> **Critical: Preserving Atom Indexing (2D vs. 3D)**
> By reconstructing the `RDKit Mol` object using canonical SMILES and embedding the conformers directly, you ensure that the atom index $i$ in the 2D graph is physically bound to the coordinate at row $i$ of each conformer. **Do not modify the molecule (e.g., re-running `Chem.MolToSmiles` or re-ordering atoms) after saving it to HDF5.** Always extract data for your model by deserializing the `mol_bytes` and iterating through the RDKit object's atoms and conformers directly. This prevents index mismatch errors that are common when 2D and 3D data are handled as separate arrays.

### Best Practices

* **Canonicalization:** Always use canonical SMILES for both the `smiles` attribute and the hash `id`. This ensures that identical chemical structures map to the same entry regardless of the input format.
* **Serialization:** Ensure that the RDKit version used to save the data matches the version used during model training to avoid potential issues with binary pickle compatibility.
* **Uniqueness:** Using the hash of the canonical SMILES as the group name effectively deduplicates your dataset, preventing the same molecule from being processed multiple times.

## Convert to Pytorch Geometric Dataset

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
