#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <omp.h>
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "CLI/CLI.hpp"
#include <unistd.h>

struct Grid {
    int nx, ny; // Grid dimension, 2d for now
    double dx, dy; // Cell size
    int ghost_size;
    int iproc, jproc;
    std::vector<double> u, u_new; // solution field

#ifdef USE_MPI_RDMA
    MPI_Win win ;
    MPI_Win win_new;
#endif USE_MPI_RDMA

    Grid(int nx_, int ny_, double dx_, double dy_, int ghost_size_, int iproc_, int jproc_)
        : 
	nx(nx_), 
	ny(ny_), 
	dx(dx_), 
	dy(dy_),
       	u    ((nx_ + 2 * ghost_size_) * (ny_ + 2 * ghost_size_), 0.0),
	u_new((nx_ + 2 * ghost_size_) * (ny_ + 2 * ghost_size_), 0.0),
        ghost_size(ghost_size_), iproc(iproc_), jproc(jproc_)
#ifdef USE_MPI_RDMA
	, win(MPI_WIN_NULL)
	, win_new(MPI_WIN_NULL)
#endif	  
	{}

#ifdef USE_MPI_RDMA
	~Grid(){
		if (win != MPI_WIN_NULL){
			MPI_Win_free(&win) ; 
		}
                if (win_new != MPI_WIN_NULL){
                        MPI_Win_free(&win_new) ;
                }		
	}
#endif 

#ifdef USE_MPI_RDMA
	void createMPIRDMAWindow(MPI_Comm comm){
		MPI_Win_create(u.data(), u.size()*sizeof(double), sizeof(double), MPI_INFO_NULL, comm, &win);
                MPI_Win_create(u_new.data(), u.size()*sizeof(double), sizeof(double), MPI_INFO_NULL, comm, &win_new);
	}
#endif

	int getIndex(int i_local, int j_local){
		return i_local + j_local * (nx + 2*ghost_size);		
	}
	double* getPtr(int i_local, int j_local){
		return &u[getIndex(i_local, j_local)];		
	}

	const double* getConstPtr(int i_local, int j_local){
		return &u[getIndex(i_local, j_local)];	
	}


};

// Stencil type
enum class StencilType { Upwind, Central };

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

void saveToText(const Grid &grid, const std::string &filename) {
    std::ofstream file(filename);
    auto ghost_size = grid.ghost_size;
    for (int j = ghost_size; j < grid.ny + ghost_size; ++j) {
        for (int i = ghost_size; i < grid.nx + ghost_size; ++i) {
            file << grid.u[i + j * (grid.nx + 2 * ghost_size) + ghost_size] << " ";
        }
        file << "\n";
    }
    file.close();
}


#ifdef USE_MPI_RDMA
void exchangeGhostCells(Grid &grid, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int north, south, east, west;
    MPI_Cart_shift(comm, 0, 1, &west, &east);
    MPI_Cart_shift(comm, 1, 1, &south, &north);

    auto ghost_size = grid.ghost_size;
    
    int stride = grid.nx + 2 * ghost_size;


    MPI_Datatype column_type ; 
    MPI_Type_vector(grid.ny, 
		    1, 
		    stride, 
		    MPI_DOUBLE, 
		    &column_type);
    MPI_Type_commit(&column_type);
    MPI_Win_fence(0, grid.win); // Assertion 0



    if (north != MPI_PROC_NULL) {
        MPI_Put(
            &grid.u[ghost_size + (ghost_size + grid.ny - 1) * stride], 
            grid.nx, MPI_DOUBLE,                                        
            north,                                           
            ghost_size + (ghost_size - 1) * stride,               
            grid.nx, MPI_DOUBLE,
            grid.win
        );
    }

    if (south != MPI_PROC_NULL) {
        MPI_Put(
            &grid.u[ghost_size + ghost_size * stride],              
            grid.nx, MPI_DOUBLE,
            south,
            ghost_size + (ghost_size + grid.ny) * stride,         
            grid.nx, MPI_DOUBLE,
            grid.win
        );
    }

    if (east != MPI_PROC_NULL) {
        MPI_Put(
            &grid.u[(ghost_size + grid.nx - 1) + ghost_size * stride], 
            1, column_type,                                          
            east,
            (ghost_size-1) + ghost_size * stride,               
            1, column_type,
            grid.win
        );
    }

    if (west != MPI_PROC_NULL) {
        MPI_Put(
            &grid.u[ghost_size + ghost_size * stride],      
            1, column_type,
            west,
            (ghost_size+grid.nx) + ghost_size * stride,       
            1, column_type,
            grid.win
        );
    }
    
    MPI_Win_fence(0, grid.win); // Assertion 0
    // Free the datatype
    MPI_Type_free(&column_type);

}

#else
void exchangeGhostCells(Grid &grid, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int north, south, east, west;
    MPI_Cart_shift(comm, 0, 1, &west, &east);
    MPI_Cart_shift(comm, 1, 1, &south, &north);

    auto ghost_size = grid.ghost_size;
    int stride = grid.nx + 2 * ghost_size;

    MPI_Datatype column_type;
    MPI_Type_vector(grid.ny,
                    1,
                    stride,
                    MPI_DOUBLE,
                    &column_type);
    MPI_Type_commit(&column_type);


    MPI_Sendrecv(
        &grid.u[ghost_size + (grid.ny + ghost_size - 1) * stride],
        grid.nx, MPI_DOUBLE, north, 0,
        &grid.u[ghost_size + (grid.ny + ghost_size) * stride],
        grid.nx, MPI_DOUBLE, north, 1,
        comm, MPI_STATUS_IGNORE
    );
    MPI_Sendrecv(
        &grid.u[ghost_size + ghost_size * stride],
        grid.nx, MPI_DOUBLE, south, 1,
        &grid.u[ghost_size + (ghost_size - 1) * stride],
        grid.nx, MPI_DOUBLE, south, 0,
        comm, MPI_STATUS_IGNORE
    );


    MPI_Sendrecv(
        &grid.u[(grid.nx + ghost_size - 1) + ghost_size * stride],
        1, column_type, east, 2,
        &grid.u[(ghost_size - 1) + ghost_size * stride],
        1, column_type, west, 2,
        comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(
        &grid.u[ghost_size + ghost_size * stride],
        1, column_type, west, 3,
        &grid.u[(grid.nx + ghost_size) + ghost_size * stride],
        1, column_type, east, 3,
        comm, MPI_STATUS_IGNORE);

    MPI_Type_free(&column_type);
}
#endif


void solveFV(Grid &grid, double velocity, StencilType stencil, int steps, int save_interval, MPI_Comm comm) {
    double dt = 0.5 * std::min(grid.dx, grid.dy) / std::abs(velocity); // Condition CFL

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto dt_dx = dt / grid.dx;
    auto dt_dy = dt / grid.dy;
    auto ghost_size = grid.ghost_size;

    int stride = grid.nx + 2 * ghost_size;

    MPI_Datatype col_type;
    MPI_Type_vector(grid.ny, 1, stride, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    int north, south, east, west;
    MPI_Cart_shift(comm, 0, 1, &west, &east);
    MPI_Cart_shift(comm, 1, 1, &south, &north);

    saveToText(grid, "frame_000.txt");


#pragma omp parallel
    for (int t = 0; t < steps; ++t) {
#pragma omp single
        {
            if (rank == 0) {
                std::cout << "\rStep: " << std::setw(6) << t << " / " << steps << std::flush;
            }

#ifdef USE_MPI_RDMA
	    // maybe we can create a win for u_new... ?
          if (grid.win != MPI_WIN_NULL) {
              MPI_Win_free(&grid.win); 
          }
          MPI_Win_create(grid.u.data(), grid.u.size() * sizeof(double), sizeof(double), MPI_INFO_NULL, comm, &grid.win);
#endif

            exchangeGhostCells(grid, comm);
        }

#pragma omp for schedule(static)
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                int idx = i + (j + ghost_size) * stride + ghost_size;
                auto pos = velocity > 0 ? true : false;
                double flux_w, flux_e, flux_s, flux_n;
                if (pos) {
                    flux_w = computeFluxPos(grid.u[idx - 1], grid.u[idx], stencil, velocity);
                    flux_e = computeFluxPos(grid.u[idx], grid.u[idx + 1], stencil, velocity);
                    flux_s = computeFluxPos(grid.u[idx - stride], grid.u[idx], stencil, velocity);
                    flux_n = computeFluxPos(grid.u[idx], grid.u[idx + stride], stencil, velocity);
                } else {
                    flux_w = computeFluxNeg(grid.u[idx - 1], grid.u[idx], stencil, velocity);
                    flux_e = computeFluxNeg(grid.u[idx], grid.u[idx + 1], stencil, velocity);
                    flux_s = computeFluxNeg(grid.u[idx - stride], grid.u[idx], stencil, velocity);
                    flux_n = computeFluxNeg(grid.u[idx], grid.u[idx + stride], stencil, velocity);
                }
                grid.u_new[idx] = grid.u[idx] - dt_dx * (flux_e - flux_w) - dt_dy * (flux_n - flux_s);
            }
        }
#pragma omp single
        {
            swap(grid.u, grid.u_new);
	    //swap(grid.win, grid.win_new) ; 

            if ((t + 1) % save_interval == 0) {
                int frame = (t + 1) / save_interval;
                std::stringstream ss;
                ss << "frame_" << std::setfill('0') << std::setw(3) << frame << "_" << grid.iproc << "_" << grid.jproc
                        << ".txt";
                saveToText(grid, ss.str());
            }
        }
    }
    if (rank == 0) {
        std::cout << std::endl;
    }
}

int main(int argc, char **argv) {
    int provided, requested = MPI_THREAD_FUNNELED;
    MPI_Init_thread(&argc, &argv, requested, &provided);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // CLI11 setup
    CLI::App app{"Finite Volume Solver"};
    int nx = 512;
    int ny = 512;
    int save_interval = 32;
    int steps = 320;
    std::string stencil_str = "upwind";

    app.add_option("--nx", nx, "Number of grid points in x direction")->check(CLI::PositiveNumber);
    app.add_option("--ny", ny, "Number of grid points in y direction")->check(CLI::PositiveNumber);
    app.add_option("--save-each-n", save_interval, "Save every n steps")->check(CLI::NonNegativeNumber);
    app.add_option("--steps", steps, "Number of simulation steps")->check(CLI::PositiveNumber);
    app.add_option("--stencil", stencil_str, "Stencil type (upwind or central)");

    CLI11_PARSE(app, argc, argv);

    StencilType stencil;
    if (stencil_str == "upwind") {
        stencil = StencilType::Upwind;
    } else if (stencil_str == "central") {
        stencil = StencilType::Central;
    } else {
        if (rank == 0)
            std::cerr << "Error: Stencil must be 'upwind' or 'central'. Using upwind as default." <<
                    std::endl;
        stencil = StencilType::Upwind;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);
    int coords[2];
    MPI_Cart_coords(comm_cart, rank, 2, coords);

    int iproc = coords[0];
    int jproc = coords[1];

    int local_nx = nx / dims[0];
    int local_ny = ny / dims[1];
    int i_offset = iproc * local_nx;
    int j_offset = jproc * local_ny;

    if (rank == 0) {
        std::cout << "local_nx: " << local_nx << " local_ny: " << local_ny << std::endl;
    }

    {
	    int ghost_size = 1;
	    double dx = 1.0 / nx, dy = 1.0 / ny;
	    Grid grid(local_nx, local_ny, dx, dy, ghost_size, iproc, jproc);
	
	
	#ifdef USE_MPI_RDMA
		grid.createMPIRDMAWindow(comm_cart)  ; 
	#endif
	
	    // Initial field remains unchanged
	    //
	
	for (int j = ghost_size; j < local_ny + ghost_size; ++j) {
	    int j_global = j_offset + (j - ghost_size);
	    for (int i = ghost_size; i < local_nx + ghost_size; ++i) {
	        int i_global = i_offset + (i - ghost_size);
	        int index = i + j * (local_nx + 2 * ghost_size); // Indice corrigé
	        if (j_global > ny / 8 && j_global < 3 * ny / 8 && i_global > nx / 8 && i_global < 3* nx / 8) {
	            grid.u[index] = 1.0;
	        } else {
	            grid.u[index] = 0.0;
	        }
	    }
	}
	/**
	#pragma omp parallel for collapse(2)
	    for (int j = ghost_size; j < local_ny + ghost_size; ++j) {
	        for (int i = ghost_size; i < local_nx + ghost_size; ++i) {
	            int index = i + j * (local_nx + 2 * ghost_size) + ghost_size;
	            if (j > local_ny / 2 && j < 3 * local_ny / 4 && i > local_nx / 4 && i < local_nx / 2) {
	                if (rank == 4) {
	                    grid.u[index] = 0.1;
	                } else {
	                    grid.u[index] = 1.0;
	                }
	            } else {
	                grid.u[index] = 0.0;
	            }
		        }
	    }
	    **/
	
	    double velocity = 1.0;
	    solveFV(grid, velocity, stencil, steps, save_interval, comm_cart);
	
    }
    if (rank == 0) {
        std::cout << "Simulation finished.\n";
        std::cout <<
                "Run 'python ../src/plot.py --folder folder --output output generate && python ../src/plot.py --folder folder animate ' to create the animation.\n";
    }

    MPI_Finalize();
    return 0;
}
