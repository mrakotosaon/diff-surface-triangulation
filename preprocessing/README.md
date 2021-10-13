# Preprocessing of Differentiable Surface Triangulation
This is our implementation the preprocessing steps for Differentiable Surface Triangulation


## Prerequisites
* libigl
* polyscope (for visualization, this is not mandatory)


## Run

To run the preprocessing:
* Copy .ply models in preprocessing_data directory and update the file shapes.txt.
* Run spectral clustering to cut individual patches
``` bash
python cut_patches.py
```
* Compute the 2D parametrization:
``` bash
python parametrize_patches.py
```

Running this code with the default parameters  might not always lead to a valid parametrization. A parametrization is valid if the 2D parametrization of patches does not have overlapping regions and the distortion is not too large. If this preprocessing leads to unvalid shapes, please consider:
1. Using smaller patches
2. Remeshing your model into faces that are more regular (close to equilateral of similar size). The isotropic_explicit_remeshing filter in MeshLab can be useful for this purpose.


## Description
We construct a 2D parametrization of patches based on LSCM. Note that any parametrization that has low distortion and no overlappings can be used in the Differentiable Surface Triangulation.
Here we generate:
*  individual meshes of the created patches (patch_*.off)
*  the ordered list of boundary vertices for each patch (boundary_*.npy)
*  the mesh representation of the patch that has been potentially cut if there were significant distortion in the original patch (cut_patch_*.npy)
* the absolute mean curvature at each vertex (mean_curvature_*.npy)
* the normal direction at each vertex (normals_*.npy)
* the 2D parametrization of each patch (param_*.npy)
* the principal curvature directions at each vertex (pc_*.npy)
