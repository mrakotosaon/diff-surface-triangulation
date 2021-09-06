# Differentiable Surface Triangulation
This is our implementation of the paper Differentiable Surface Triangulation that enables optimization for any per-vertex or per-face differentiable objective function over the space of underlying surface triangulations.


![diff-surface-triangulation](img/diff_surface_triangulation.png "Differentiable Surface Triangulation")


This code was written by [Marie-Julie Rakotosaona](http://www.lix.polytechnique.fr/Labo/Marie-Julie.RAKOTOSAONA/).

## Prerequisites
* CUDA and CuDNN (changing the code to run on CPU should require few changes)
* Python 3.6
* Tensorflow 1.15

## Setup
Install required python packages, if they are not already installed:
``` bash
pip install numpy
pip install scipy
pip install trimesh
```


Clone this repository:
``` bash
git clone https://github.com/mrakotosaon/diff-meshing.git
cd dse-meshing
```

## Optimization

To run the optimization on the given pre-processed patches for the curvature alignment experiment:
``` bash
python optimize_curvature.py
```
for the triangle size experiment:
``` bash
python optimize_triangle_size.py
```





## Citation
If you use our work, please cite our paper.


[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us.
