#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <unistd.h>


struct Grid {
    int nx, ny; // Grid dimension, 2d for now
    double dx, dy; // Cell size
    int ghost_size;
    std::vector<double> u; // solution field

    Grid(int nx_, int ny_, double dx_, double dy_, int ghost_size_)
        : nx(nx_), ny(ny_), dx(dx_), dy(dy_), u((nx_ + 2 * ghost_size_) * (ny_ + 2 * ghost_size_), 0.0),
          ghost_size(ghost_size_) {
    }
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
void saveToText(const Grid &grid, const std::string &filename) {
    std::ofstream file(filename);
    auto ghost_size = grid.ghost_size;
    for (int j = ghost_size; j < grid.ny + ghost_size; ++j) {
        for (int i = ghost_size; i < grid.nx + ghost_size; ++i) {
            file << grid.u[i + j * (grid.nx+2*ghost_size) + ghost_size] << " ";
        }
        file << "\n";
    }
    file.close();
}

// Finit volume solver
void solveFV(Grid &grid, double velocity, StencilType stencil, int steps, int save_interval, MPI_Comm comm) {
    std::vector<double> u_new(grid.u.size());
    double dt = 0.5 * std::min(grid.dx, grid.dy) / std::abs(velocity); // Condition CFL

    // Save init state
    saveToText(grid, "frame_000.txt");

    auto dt_dx = dt / grid.dx;
    auto dt_dy = dt / grid.dy;
    auto ghost_size = grid.ghost_size;

    int stride = grid.nx + 2 * ghost_size;

    // Création du type colonne pour les échanges MPI
    MPI_Datatype col_type;
    MPI_Type_vector(grid.ny, 1, stride, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    // Identification des voisins
    int north, south, east, west;
    MPI_Cart_shift(comm, 0, 1, &west, &east); // Direction x
    MPI_Cart_shift(comm, 1, 1, &south, &north); // Direction y

#pragma omp parallel
    for (int t = 0; t < steps; ++t) {
#pragma omp single
        {
            // Échange nord-sud
            MPI_Sendrecv(&grid.u[ghost_size + grid.ny * stride], grid.nx, MPI_DOUBLE, north, 0,
                         &grid.u[ghost_size + 0 * stride], grid.nx, MPI_DOUBLE, south, 0, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&grid.u[ghost_size + 1 * stride], grid.nx, MPI_DOUBLE, south, 1,
                         &grid.u[ghost_size + (grid.ny + 1) * stride], grid.nx, MPI_DOUBLE, north, 1, comm, MPI_STATUS_IGNORE);

            // Échange est-ouest
            MPI_Sendrecv(&grid.u[grid.nx + ghost_size * stride], 1, col_type, east, 2,
                         &grid.u[0 + ghost_size * stride], 1, col_type, west, 2, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&grid.u[1 + ghost_size * stride], 1, col_type, west, 3,
                         &grid.u[(grid.nx + 1) + ghost_size * stride], 1, col_type, east, 3, comm, MPI_STATUS_IGNORE);
        }

#pragma omp for  schedule(dynamic)
        for (int j = ghost_size; j < grid.ny + ghost_size; ++j) {
            for (int i = ghost_size; i < grid.nx + ghost_size; ++i) {
                int idx = i + j * (grid.nx + 2 * ghost_size) + ghost_size;
                auto pos = velocity > 0 ? true : false;
                double flux_w, flux_e, flux_s, flux_n;
                if (pos) {
                    flux_w = computeFluxPos(grid.u[idx - 1], grid.u[idx], stencil, velocity);
                    flux_e = computeFluxPos(grid.u[idx], grid.u[idx + 1], stencil, velocity);
                    flux_s = computeFluxPos(grid.u[idx - grid.nx], grid.u[idx], stencil, velocity);
                    flux_n = computeFluxPos(grid.u[idx], grid.u[idx + grid.nx], stencil, velocity);
                } else {
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
                int rank = -1 ;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                int frame = (t + 1) / save_interval;
                std::stringstream ss;
                ss << "frame_" << std::setfill('0') << std::setw(3) << frame << "_" << rank << ".txt";
                saveToText(grid, ss.str());
            }
        }
    }
}

int main(int argc, char **argv) {
    int provided, requested = MPI_THREAD_FUNNELED;
    MPI_Init_thread(&argc, &argv, requested, &provided);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Creation d'un Cartesien MPI
    int dims[2] = {0,0} ;
    MPI_Dims_create(size, 2, dims) ;
    int periods[2] = {0,0} ;
    MPI_Comm comm_cart ;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);
    int coords[2] ;
    MPI_Cart_coords(comm_cart, rank, 2, coords);

    int iproc, jproc ;
    iproc = coords[0] ;
    jproc = coords[1] ;


    int global_nx = 4096 ;
    int global_ny = 4096 ;
    int local_nx = global_nx / dims[0] ;
    int local_ny = global_ny / dims[1] ;
    int i_offset = iproc * local_nx ;
    int j_offset = jproc * local_ny ;

    std::cout << " local_nx : " << local_nx << " local_ny : " << local_ny << std::endl ;

    // Init of the computation

    int ghost_size = 1;
    double dx = 1.0 / global_nx, dy = 1.0 / global_ny;
    Grid grid(local_nx, local_ny, dx, dy, ghost_size);

    // Initial field
#pragma omp parallel for collapse(2)
    for (int j = ghost_size; j < local_ny + ghost_size; ++j) {
        for (int i = ghost_size; i < local_nx + ghost_size; ++i) {
            double x = i * (dx+2*ghost_size), y = j * (dy+2*ghost_size);
            int index = i + j * (local_nx+2*ghost_size) + ghost_size;
            //            grid.u[i + j * nx] = std::exp(-50 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)));
            if (j > local_ny / 4 && j < local_ny / 2 && i > local_nx / 4 && i < local_nx / 2) {
                grid.u[index] = 1.0;
            } else {
                grid.u[index] = 0.0;
            }
        }
    }

    // Compute loop
    double velocity = 1.0;
    StencilType stencil = StencilType::Upwind;
    //    StencilType stencil = StencilType::Central;
    int steps = 2000;
    int save_interval = 100;
    solveFV(grid, velocity, stencil, steps, save_interval, comm_cart);

    std::cout << "Simulation finished. The frames are saved from frame_000.txt to frame_100.txt.\n";
    std::cout << "Run 'gnuplot animate.gp' to create the animation.\n";


    MPI_Finalize();
    return 0;
}
