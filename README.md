# FastTrack

## Introduction
we have demonstrated a fast, accurate, and flexible framework for predicting atomic diffusion barriers in crystalline solids by integrating universal machine‐learning force fields with three‐dimensional potential‐energy‐surface sampling and interpolation.


## Download / Installation

Install from source:
```bash
git clone https://github.com/atomly-materials-research-lab/FastTrack.git
cd FastTrack
vim FastTrack/config.py
pip install .
```
Set your ML force field parameters in config.py

(Optional) Development install:
```bash
pip install -e  .
```

## Usage Example
A minimal example to get users started quickly.

Specify the machine learning force field in FastTrack/config.py, including the model and parameter paths.

```python
from FastTrack import kkk  

barrier_energy = kkk("LiFePO4.cif",'Li',1)   #maximum lithiation limit
#or
barrier_energy = kkk("LiFePO4.cif",'Li',0)   #maximum delithiation limit
```


## Citation
If you use this repository in your research, please cite the original work:

```bibtex
@article{Kang2025FastTrack,
  title = {FastTrack: A Fast Method to Evaluate Mass Transport in Solid Leveraging Universal Machine Learning Interatomic Potential},
  author = {Kang, Hanwen and Lu, Tenglong and Qi, Zhanbin and Guo, Jiandong and Meng, Sheng and Liu, Miao},
  journal = {arXiv preprint arXiv:2508.10505},
  year = {2025},
  doi = {10.48550/arXiv.2508.10505},
  url = {https://arxiv.org/abs/2508.10505}
}

