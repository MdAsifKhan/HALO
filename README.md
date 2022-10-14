# HALO

**HALO:  Hamiltonian latent operators for content and motion disentanglement in image sequences.**

This repo provides the official implementation of [HALO](https://arxiv.org/abs/2112.01641)


# [HALO Framework] (./figures/flowdiag.pdf)


# Setup Conda Environment

```
conda env create -f environment.yaml
conda activate HALO
```


# Training Setup
```
python train.py --config configs/sprites.yaml
```


# Testing
```
python test.py --config configs/sprites.yaml
```


# Code update in progress (the full pipeline will be updated soon)


## Cite us
```bibtex
@article{khan2021hamiltonian,
  title={Hamiltonian latent operators for content and motion disentanglement in image sequences},
  author={Khan, Asif and Storkey, Amos},
  journal={arXiv preprint arXiv:2112.01641},
  year={2021}
}

```