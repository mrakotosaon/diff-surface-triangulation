#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/polygon_soup_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/timing.h"

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "args/args.hxx"
#include "imgui.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

// == Geometry-central data
std::unique_ptr<ManifoldSurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;

// Polyscope visualization handle, to quickly add data to the surface

// An edge network while processing flips
std::unique_ptr<FlipEdgeNetwork> edgeNetwork;

// boundary 2500 6704 2601 2645 6642

int main(int argc, char** argv) {
    std::cout << "we are here " << '\n';
    std::cout << "boundary file "<< argv[1] << '\n';
    std::cout << "output file "<< argv[2] << '\n';
    std::cout << "input file "<< argv[3] << '\n';
    //  ./bin/flip_geodesics ../src/boundary.txt my_mesh.obj ../lion-head.off
    // upload boundary
    std::vector<std::vector<int>> v;
    std::ifstream ifs(argv[1]);
    std::string tempstr;
    int tempint;
    char delimiter;

    while (std::getline(ifs, tempstr)) {
        std::istringstream iss(tempstr);
        std::vector<int> tempv;
        while (iss >> tempint) {
            tempv.push_back(tempint);
        }
        v.push_back(tempv);
    }

    //for (auto row : v) {
    //  }
    std::tie(mesh, geometry) = readManifoldSurfaceMesh(argv[3]);
    writeSurfaceMesh(*mesh, *geometry, argv[2]);
    for (auto row : v){
      std::cout << row[0] <<" "<< row[1]<< '\n';
      std::tie(mesh, geometry) = readManifoldSurfaceMesh(argv[2]);
      std::cout << row[0] <<' ' << row[1]<< '\n';
      Vertex vStart = mesh->vertex(row[0]);
      Vertex vEnd = mesh->vertex(row[1]);
      edgeNetwork = FlipEdgeNetwork::constructFromDijkstraPath(*mesh, *geometry, vStart, vEnd);
      edgeNetwork->iterativeShorten();
      edgeNetwork->posGeom = geometry.get();
      writeSurfaceMesh(*edgeNetwork->tri->intrinsicMesh, *geometry, argv[2]);
    }


    return EXIT_SUCCESS;
}
