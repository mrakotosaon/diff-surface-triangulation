# Postprocessing of Differentiable Surface Triangulation
This is our implementation the postprocessing steps for Differentiable Surface Triangulation. The code relies on the excellent implementation of ["You Can Find Geodesic Paths in Triangle Meshes by Just Flipping Edges"](https://nmwsharp.com/research/flip-geodesics/), [on github](https://github.com/nmwsharp/flip-geodesics-demo).


## Setup

``` bash
cd flip-geodesic-demo
mkdir build
cd build
cmake ..
make
```

## Run

* Run the post-processing on optimized patches
``` bash
python cut_boundary.py
```
