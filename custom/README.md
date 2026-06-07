# Pre-Training with Custom Dataset

## Custom Dataset Preparation

See [Sample](./dataset/sample/raw/build_sample_dataset.ipynb).

## Convert to Pytorch Geometric Dataset

After prepare `dataset/dataset_name/raw/dataset.h5`, dataset for GraphMVP is build with the following:

```python
python custom_dataset.py --n_mol 50 --n_conf 5 --n_upper 1000 --data_folder dataset/dataset_name

```

## Run

TODO: Future edit.

```python
python pretrain_GraphMVP_hybrid.py --epochs=100 --dataset=GEOM_3D_nmol50000_nconf5_nupper1000_morefeat --batch_size=256 --SSL_masking_ratio=0.15 --CL_similarity_metric=EBM_dot_prod --T=0.1 --normalize --AE_model=VAE --AE_loss=l2 --detach_target --beta=1 --alpha_1=1 --alpha_2=1 --SSL_2D_mode=CP --alpha_3=1 --num_interactions=6 --num_gaussians=51 --cutoff=10 --schnet_lr_scale=0.1 --dropout_ratio=0 --num_workers=8 --output_model_dir=../output/GraphMVP_Hybrid_C/pretraining
```
