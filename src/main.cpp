#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include <hdf5.h>
#include <fstream>
//#include <H5Cpp.h>
#include <highfive/H5File.hpp>

//using namespace H5;
using namespace HighFive;

// Grid structure
struct Grid {
    int nx, ny;              // Grid dimensions
    double dx, dy;           // Cell sizes
    std::vector<double> u;   // Solution field

    Grid(int nx_, int ny_, double dx_, double dy_)
        : nx(nx_), ny(ny_), dx(dx_), dy(dy_), u(nx_ * ny_, 0.0) {}
};

// Stencil types
enum class StencilType { Upwind, Central };

// Compute flux based on stencil
double computeFlux(double u_left, double u_right, StencilType stencil, double velocity) {
    switch (stencil) {
        case StencilType::Upwind:
            return velocity > 0 ? velocity * u_left : velocity * u_right;
        case StencilType::Central:
            return velocity * (u_left + u_right) / 2.0;
        default:
            return 0.0;
    }
}

// Finite volume solver
void solveFV(Grid& grid, double velocity, StencilType stencil, int steps) {
    std::vector<double> u_new(grid.u.size());
    double dt = 0.1 * std::min(grid.dx, grid.dy) / std::abs(velocity); // CFL condition

    for (int t = 0; t < steps; ++t) {
        #pragma omp parallel for collapse(2)
        for (int j = 1; j < grid.ny - 1; ++j) {
            for (int i = 1; i < grid.nx - 1; ++i) {
                int idx = i + j * grid.nx;
                double flux_w = computeFlux(grid.u[idx - 1], grid.u[idx], stencil, velocity);
                double flux_e = computeFlux(grid.u[idx], grid.u[idx + 1], stencil, velocity);
                double flux_s = computeFlux(grid.u[idx - grid.nx], grid.u[idx], stencil, velocity);
                double flux_n = computeFlux(grid.u[idx], grid.u[idx + grid.nx], stencil, velocity);

                u_new[idx] = grid.u[idx] - dt / grid.dx * (flux_e - flux_w) - dt / grid.dy * (flux_n - flux_s);
            }
        }
        grid.u = u_new; // Update solution
    }
}

// Save to HDF5/XDMF
void saveToHDF5(const Grid& grid, const std::string& filename) {
	// TODO : DO not copy.
	// but for now it is like that
    auto nx = grid.nx ; 
    auto ny = grid.ny ;
    auto vec1D = grid.u;
    std::vector<std::vector<double>> vec2D(nx, std::vector<double>(ny));
    // Copie des éléments du vecteur 1D dans le vecteur 2D
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            vec2D[i][j] = vec1D[i * ny + j];
        }
    }



    File file(filename + ".h5", File::Overwrite);
    std::vector<size_t> dims = { static_cast<size_t>(grid.ny), static_cast<size_t>(grid.nx) };
    DataSet dataset = file.createDataSet<double>("u", DataSpace(dims));
	dataset.write(vec2D) ; 


    std::ofstream xdmfFile(filename + ".xdmf");
    xdmfFile << R"(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="StructuredGrid" GridType="Uniform">
      <Topology TopologyType="2DSMESH" Dimensions=")" << ny << " " << nx << R"(" />
      <Geometry GeometryType="XY">
        <DataItem Format="XML" NumberType="Float" Dimensions=")" << ny << " " << nx << R"( 2">
          <!-- Placeholder for coordinates -->
        </DataItem>
      </Geometry>
      <Attribute Name="u" AttributeType="Scalar" Center="Cell">
        <DataItem Format="HDF" NumberType="Float" Dimensions=")" << ny << " " << nx << R"(">
          )" << filename << ".h5:/u" << R"(
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
)";
    xdmfFile.close();

}

int main() {
    // Initialize grid
    int nx = 1000, ny = 1000;
    double dx = 1.0 / (nx - 1), dy = 1.0 / (ny - 1);
    Grid grid(nx, ny, dx, dy);

    // Initial condition (e.g., a pulse)
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = i * dx, y = j * dy;
            grid.u[i + j * nx] = std::exp(-50 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)));
        }
    }

    // Solve
    double velocity = 1.0;
    StencilType stencil = StencilType::Upwind;
    int steps = 1000;
    solveFV(grid, velocity, stencil, steps);

    // Save result
    saveToHDF5(grid, "output");

    std::cout << "Simulation completed. Results saved to output.h5 and output.xdmf.\n";
    return 0;
}
