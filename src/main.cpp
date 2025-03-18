#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <iomanip>

struct Grid {
    int nx, ny;              // Grid dimension, 2d for now
    double dx, dy;           // Cell size
    std::vector<double> u;   // solution field 

    Grid(int nx_, int ny_, double dx_, double dy_)
        : nx(nx_), ny(ny_), dx(dx_), dy(dy_), u(nx_ * ny_, 0.0) {}
};

// Stencil type
enum class StencilType { Upwind, Central };

// Flux 
double computeFluxNeg(double u_left, double u_right, StencilType stencil, double velocity) {
    switch (stencil) {
        case StencilType::Upwind:
            return velocity * u_right;
        case StencilType::Central:
            return velocity * (u_left + u_right) / 2.0;
        default:
            return 0.0;
    }
}

// Flux 
double computeFluxPos(double u_left, double u_right, StencilType stencil, double velocity) {
    switch (stencil) {
        case StencilType::Upwind:
            return velocity * u_left;
        case StencilType::Central:
            return velocity * (u_left + u_right) / 2.0;
        default:
            return 0.0;
    }
}


// GNUPLOT saving
void saveToText(const Grid& grid, const std::string& filename) {
    std::ofstream file(filename);
    for (int j = 0; j < grid.ny; ++j) {
        for (int i = 0; i < grid.nx; ++i) {
            file << grid.u[i + j * grid.nx] << " ";
        }
        file << "\n";
    }
    file.close();
}

// Finit volume solver
void solveFV(Grid& grid, double velocity, StencilType stencil, int steps, int save_interval) {
    std::vector<double> u_new(grid.u.size());
    double dt = 0.5 * std::min(grid.dx, grid.dy) / std::abs(velocity); // Condition CFL

    // Save init state
    saveToText(grid, "frame_000.txt");

    auto dt_dx = dt / grid.dx;
    auto dt_dy = dt / grid.dy ;
    #pragma omp parallel
    for (int t = 0; t < steps; ++t) {
        #pragma omp for  schedule(dynamic)
        for (int j = 1; j < grid.ny - 1; ++j) {
            for (int i = 1; i < grid.nx - 1; ++i) {
                int idx = i + j * grid.nx;
		auto pos = velocity > 0 ? true : false ;
		double flux_w, flux_e, flux_s, flux_n ; 
		if (pos){
	                flux_w = computeFluxPos(grid.u[idx - 1], grid.u[idx], stencil, velocity);
	                flux_e = computeFluxPos(grid.u[idx], grid.u[idx + 1], stencil, velocity);
	                flux_s = computeFluxPos(grid.u[idx - grid.nx], grid.u[idx], stencil, velocity);
	                flux_n = computeFluxPos(grid.u[idx], grid.u[idx + grid.nx], stencil, velocity);
		}
		else{
	                flux_w = computeFluxNeg(grid.u[idx - 1], grid.u[idx], stencil, velocity);
	                flux_e = computeFluxNeg(grid.u[idx], grid.u[idx + 1], stencil, velocity);
	                flux_s = computeFluxNeg(grid.u[idx - grid.nx], grid.u[idx], stencil, velocity);
	                flux_n = computeFluxNeg(grid.u[idx], grid.u[idx + grid.nx], stencil, velocity);
		}
                u_new[idx] = grid.u[idx] - dt_dx * (flux_e - flux_w) - dt_dy * (flux_n - flux_s);
            }
        }
	#pragma omp single
	{
	swap(grid.u, u_new);

        // Saving data
        if ((t + 1) % save_interval == 0) {
            int frame = (t + 1) / save_interval;
            std::stringstream ss;
            ss << "frame_" << std::setfill('0') << std::setw(3) << frame << ".txt";
            saveToText(grid, ss.str());
        }
	}
    }
}

int main() {
    // Init of the computation
    int nx = 4096, ny = 4096;
    double dx = 1.0 / (nx - 1), dy = 1.0 / (ny - 1);
    Grid grid(nx, ny, dx, dy);

    // Initial field
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double x = i * dx, y = j * dy;
	    int index = i + j * nx;
//            grid.u[i + j * nx] = std::exp(-50 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)));
	    if (j > ny / 4 && j < ny / 2 && i > nx / 4 && i < nx / 2){
		    grid.u[index] = 1.0;
	    }
	    else{
		grid.u[index] = 0.0 ; 
	    }
        }
    }

    // Compute loop
    double velocity = 1.0;
    StencilType stencil = StencilType::Upwind;
//    StencilType stencil = StencilType::Central;
    int steps = 50;
    int save_interval = 200;
    solveFV(grid, velocity, stencil, steps, save_interval);

    std::cout << "Simulation finished. The frames are saved from frame_000.txt to frame_100.txt.\n";  
    std::cout << "Run 'gnuplot animate.gp' to create the animation.\n";


    return 0;
}
