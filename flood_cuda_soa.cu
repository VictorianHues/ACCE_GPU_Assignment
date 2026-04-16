/*
 * NOTE: READ CAREFULLY
 * Here the function `do_compute` is just a copy of the CPU sequential version.
 * Implement your GPU code with CUDA here. Check the README for further instructions.
 * You can modify everything in this file, as long as we can compile the executable using
 * this source code, and Makefile.
 *
 * Simulation of rainwater flooding
 * CUDA version (Implement your parallel version here)
 *
 * Adapted for ACCE at the VU, Period 5 2025-2026 from the original version by
 * Based on the EduHPC 2025: Peachy assignment, Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2024/2025
 */

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Headers for the CUDA assignment versions */
#include <cuda.h>

/* Example of macros for error checking in CUDA */
#define CUDA_CHECK_FUNCTION(call)                                                                                      \
    {                                                                                                                  \
        cudaError_t check = call;                                                                                      \
        if (check != cudaSuccess)                                                                                      \
            fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check));                      \
    }
#define CUDA_CHECK_KERNEL()                                                                                            \
    {                                                                                                                  \
        cudaError_t check = cudaGetLastError();                                                                        \
        if (check != cudaSuccess)                                                                                      \
            fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check));               \
    }

/*
 * Utils: Random generator
 */
#include "rng.c"

/*
 * Header file: Contains constants and definitions
 */
#include "flood.h"

extern "C" double get_time();

static constexpr int RAIN_CONST_MAX_CLOUDS = 1024;
static constexpr int RAIN_CONST_FASTPATH_CLOUDS = 256;

__constant__ float c_rain_cloud_x[RAIN_CONST_MAX_CLOUDS];
__constant__ float c_rain_cloud_y[RAIN_CONST_MAX_CLOUDS];
__constant__ float c_rain_cloud_radius[RAIN_CONST_MAX_CLOUDS];
__constant__ float c_rain_cloud_intensity[RAIN_CONST_MAX_CLOUDS];
__constant__ float c_rain_cloud_sqrt_divr[RAIN_CONST_MAX_CLOUDS];
__constant__ int c_rain_cloud_active[RAIN_CONST_MAX_CLOUDS];

__device__ __forceinline__ unsigned long long warp_reduce_sum_ull(unsigned long long value) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    return value;
}

/*
 *  - one thread per cell
 *  - initialize per-cell state arrays
 *  - initialize the 3D neighbor contribution buffer for that cell
 */
__global__ void init_state_kernel(int rows, int columns, int *d_water_level, float *d_spillage_flag,
                                  float *d_spillage_level, float *d_spillage_from_neigh) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= columns)
        return;

    int idx = row * columns + col;
    d_water_level[idx] = 0;
    d_spillage_flag[idx] = 0.0f;
    d_spillage_level[idx] = 0.0f;

    int base = idx * CONTIGUOUS_CELLS;
    for (int depth = 0; depth < CONTIGUOUS_CELLS; depth++) {
        d_spillage_from_neigh[base + depth] = 0.0f;
    }
}

__global__ void cloud_movement_kernel_soa(int num_clouds, float *d_cloud_x, float *d_cloud_y,
                                          const float *d_cloud_speed, const float *d_cloud_angle,
                                          const int *d_cloud_active) {
    int cloud_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cloud_idx >= num_clouds)
        return;

    if (d_cloud_active[cloud_idx] == 0)
        return;

    float km_minute = d_cloud_speed[cloud_idx] / 60.0f;
    float angle_rad = d_cloud_angle[cloud_idx] * (float)M_PI / 180.0f;
    d_cloud_x[cloud_idx] += km_minute * cosf(angle_rad);
    d_cloud_y[cloud_idx] += km_minute * sinf(angle_rad);
}

__global__ void alt_calc_rainfall_kernel_soa(int rows, int columns, int num_clouds,
                                              unsigned long long *d_total_rainfall,
                                              const float *__restrict__ d_cloud_x,
                                              const float *__restrict__ d_cloud_y,
                                              const float *__restrict__ d_cloud_radius,
                                              const float *__restrict__ d_cloud_intensity,
                                              const float *__restrict__ d_cloud_sqrt_divr_intensity,
                                              const int *__restrict__ d_cloud_active, float ex_factor,
                                              int *d_water_level) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    int in_bounds = (row < rows && col < columns);
    float x_pos = 0.0f;
    float y_pos = 0.0f;
    if (in_bounds) {
        x_pos = COORD_MAT2SCEN_X(col);
        y_pos = COORD_MAT2SCEN_Y(row);
    }

    extern __shared__ unsigned char shared_raw[];
    float *shared_cloud_x = (float *)shared_raw;
    float *shared_cloud_y = shared_cloud_x + threads_per_block;
    float *shared_cloud_radius = shared_cloud_y + threads_per_block;
    float *shared_cloud_intensity = shared_cloud_radius + threads_per_block;
    float *shared_cloud_sqrt_divr = shared_cloud_intensity + threads_per_block;
    int *shared_cloud_active = (int *)(shared_cloud_sqrt_divr + threads_per_block);

    float cell_rainfall = 0.0f;
    float rain_scale = ex_factor / 60000.0f; // cm/h -> m/min

    for (int base = 0; base < num_clouds; base += threads_per_block) {
        int cloud_idx = base + tid;
        if (cloud_idx < num_clouds) {
            shared_cloud_x[tid] = d_cloud_x[cloud_idx];
            shared_cloud_y[tid] = d_cloud_y[cloud_idx];
            shared_cloud_radius[tid] = d_cloud_radius[cloud_idx];
            shared_cloud_intensity[tid] = d_cloud_intensity[cloud_idx];
            shared_cloud_sqrt_divr[tid] = d_cloud_sqrt_divr_intensity[cloud_idx];
            shared_cloud_active[tid] = d_cloud_active[cloud_idx];
        } else {
            shared_cloud_active[tid] = 0;
        }
        __syncthreads();

        int tile_clouds = MIN(threads_per_block, num_clouds - base);
        if (in_bounds) {
            for (int cloud = 0; cloud < tile_clouds; cloud++) {
                if (shared_cloud_active[cloud] == 0)
                    continue;

                float dx = x_pos - shared_cloud_x[cloud];
                float dy = y_pos - shared_cloud_y[cloud];
                float dist2 = dx * dx + dy * dy;
                float cloud_radius = shared_cloud_radius[cloud];
                float radius2 = cloud_radius * cloud_radius;

                if (dist2 < radius2) {
                    float distance = sqrtf(dist2);
                    float rain =
                        fmaxf(0.0f, shared_cloud_intensity[cloud] - distance * shared_cloud_sqrt_divr[cloud]);
                    cell_rainfall += rain_scale * rain;
                }
            }
        }

        __syncthreads();
    }

    unsigned long long fixed_rain = 0;
    if (in_bounds) {
        fixed_rain = (unsigned long long)FIXED(cell_rainfall);
        accessMat(d_water_level, row, col) += (int)fixed_rain;
    }

    // Aggregate per-warp contributions before global atomic update.
    unsigned long long warp_sum = warp_reduce_sum_ull(fixed_rain);
    if ((tid & (warpSize - 1)) == 0 && warp_sum > 0)
        atomicAdd(d_total_rainfall, warp_sum);
}

__global__ void cloud_movement_kernel(int num_clouds, Cloud_t *d_clouds) {
    int cloud_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cloud_idx >= num_clouds)
        return;

    Cloud_t *c_cloud = &d_clouds[cloud_idx];
    float km_minute = c_cloud->speed / 60;
    c_cloud->x += km_minute * cos(c_cloud->angle * M_PI / 180.0);
    c_cloud->y += km_minute * sin(c_cloud->angle * M_PI / 180.0);
}

__global__ void rainfall_kernel(int num_clouds, unsigned long long *d_total_rainfall, Cloud_t *d_clouds, int rows, int columns, float ex_factor,
                                int *d_water_level) {
    int cloud_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (cloud_idx >= num_clouds)
        return;

    Cloud_t c_cloud = d_clouds[cloud_idx];
    // Compute the bounding box area of the cloud
    float row_start = COORD_SCEN2MAT_Y(MAX(0, c_cloud.y - c_cloud.radius));
    float row_end = COORD_SCEN2MAT_Y(MIN(c_cloud.y + c_cloud.radius, SCENARIO_SIZE));
    float col_start = COORD_SCEN2MAT_X(MAX(0, c_cloud.x - c_cloud.radius));
    float col_end = COORD_SCEN2MAT_X(MIN(c_cloud.x + c_cloud.radius, SCENARIO_SIZE));
    float distance;

    // Add rain to the ground water level
    for (float row_pos = row_start; row_pos < row_end; row_pos++) {
        for (float col_pos = col_start; col_pos < col_end; col_pos++) {
            float x_pos = COORD_MAT2SCEN_X(col_pos);
            float y_pos = COORD_MAT2SCEN_Y(row_pos);
            distance =
                sqrt((x_pos - c_cloud.x) * (x_pos - c_cloud.x) + (y_pos - c_cloud.y) * (y_pos - c_cloud.y));
            if (distance < c_cloud.radius) {
                float rain = ex_factor *
                             MAX(0, c_cloud.intensity - distance / c_cloud.radius * sqrt(c_cloud.intensity));
                float meters_per_minute = rain / 1000 / 60;
                accessMat(d_water_level, (int)row_pos, (int)col_pos) += FIXED(meters_per_minute);
                atomicAdd(d_total_rainfall, (unsigned long long)FIXED(meters_per_minute));
            }
        }
    }
}

__global__ void alternative_rainfall_kernel(int rows, int columns, int num_clouds, 
                                            unsigned long long *d_total_rainfall, Cloud_t *d_clouds, 
                                            float ex_factor, int *d_water_level) {
    /* Iterate through matrix instead of clouds */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= columns)
        return;

    float x_pos = COORD_MAT2SCEN_X(col);
    float y_pos = COORD_MAT2SCEN_Y(row);

    for (int cloud = 0; cloud < num_clouds; cloud++) {
        Cloud_t c_cloud = d_clouds[cloud];

        float distance =
            sqrt(
                (x_pos - c_cloud.x) * (x_pos - c_cloud.x) + (y_pos - c_cloud.y) * (y_pos - c_cloud.y)
            );
        if (distance < c_cloud.radius) {
            /*
            * The rainfall contribution of a cloud is computed as:
            * rain = ex_factor * max(0, c_cloud.intensity - distance / c_cloud.radius * sqrt(c_cloud.intensity))
            * r = ex * max(0, I - (d / R) * sqrt(I))
            */
            float rain = ex_factor *
                         MAX(0, c_cloud.intensity - distance / c_cloud.radius * sqrt(c_cloud.intensity));
            float meters_per_minute = rain / 1000 / 60;
            accessMat(d_water_level, row, col) += FIXED(meters_per_minute);
            atomicAdd(d_total_rainfall, (unsigned long long)FIXED(meters_per_minute));
        }  
    }
}

__global__ void alt_calc_rainfall_kernel(int rows, int columns, int num_clouds, 
                                            unsigned long long *d_total_rainfall, Cloud_t *d_clouds, 
                                            float ex_factor, int *d_water_level) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= columns)
        return;

    float x_pos = COORD_MAT2SCEN_X(col);
    float y_pos = COORD_MAT2SCEN_Y(row);

    float cell_rainfall = 0;

    for (int cloud = 0; cloud < num_clouds; cloud++) {
        Cloud_t c_cloud = d_clouds[cloud];
        float cloud_x = c_cloud.x;
        float cloud_y = c_cloud.y;
        float cloud_radius = c_cloud.radius;
        float cloud_intensity = c_cloud.intensity;

        float distance =
            sqrt(
                (x_pos - cloud_x) * (x_pos - cloud_x) + (y_pos - cloud_y) * (y_pos - cloud_y)
            );
        if (distance < cloud_radius) {
            /*
            * The rainfall contribution of a cloud is computed as:
            * rain = ex_factor * max(0, cloud_intensity - distance / cloud_radius * sqrt(cloud_intensity))
            * r = ex * max(0, I - (d / R) * sqrt(I))
            */
            float rain = ex_factor *
                        MAX(0, cloud_intensity - distance / cloud_radius * sqrt(cloud_intensity));
            float meters_per_minute = rain / 1000 / 60;
            cell_rainfall += meters_per_minute;
        }
    }
    accessMat(d_water_level, row, col) += FIXED(cell_rainfall);
    atomicAdd(d_total_rainfall, (unsigned long long)FIXED(cell_rainfall));
}

__global__ void compute_spillage_kernel(int rows, int columns, unsigned long long *d_total_water_loss, float *d_ground, 
                                        int *d_water_level, float *d_spillage_flag,
                                        float *d_spillage_level, float *d_spillage_from_neigh) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int cell_pos, new_row, new_col;

    if (row >= rows || col >= columns)
        return;
        
    if (accessMat(d_water_level, row, col) > 0) {
        float sum_diff = 0;
        float my_spillage_level = 0;

        /* Differences between current-cell level and its neighbours  */
        float current_height =
            accessMat(d_ground, row, col) + FLOATING(accessMat(d_water_level, row, col));

        // Iterate over the four neighboring cells using the displacement array
        for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
            new_row = row + displacements[cell_pos][0];
            new_col = col + displacements[cell_pos][1];

            float neighbor_height;

            // Check if the new position is within the matrix boundaries
            if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns)
                // Out of borders: Same height as the cell with no water
                neighbor_height = accessMat(d_ground, row, col);
            else
                // Neighbor cell: Ground height + water level
                neighbor_height = accessMat(d_ground, new_row, new_col) +
                                    FLOATING(accessMat(d_water_level, new_row, new_col));

            // Compute level differences
            if (current_height >= neighbor_height) {
                float height_diff = current_height - neighbor_height;
                sum_diff += height_diff;
                my_spillage_level = MAX(my_spillage_level, height_diff);
            }
        }
        my_spillage_level = MIN(FLOATING(accessMat(d_water_level, row, col)), my_spillage_level);

        // Compute proportion of spillage to each neighbor
        if (sum_diff > 0.0) {
            float proportion = my_spillage_level / sum_diff;
            // If proportion is significative, spillage
            if (proportion > 1e-8) {
                accessMat(d_spillage_flag, row, col) = 1;
                accessMat(d_spillage_level, row, col) = my_spillage_level;

                // Iterate over the four neighboring cells using the displacement array
                for (cell_pos = 0; cell_pos < 4; cell_pos++) {
                    new_row = row + displacements[cell_pos][0];
                    new_col = col + displacements[cell_pos][1];

                    float neighbor_height;

                    // Check if the new position is within the matrix boundaries
                    // Race condition
                    if (new_row < 0 || new_row >= rows || new_col < 0 || new_col >= columns) {
                        // Spillage out of the borders: Water loss
                        neighbor_height = accessMat(d_ground, row, col);
                        if (current_height >= neighbor_height) {
                            atomicAdd(d_total_water_loss,
                                      (unsigned long long)FIXED(proportion * (current_height - neighbor_height) / 2));
                        }
                    } else {
                        // Spillage to a neighbor cell
                        neighbor_height = accessMat(d_ground, new_row, new_col) +
                                            FLOATING(accessMat(d_water_level, new_row, new_col));
                        if (current_height >= neighbor_height) {
                            int depths = CONTIGUOUS_CELLS;
                            accessMat3D(d_spillage_from_neigh, new_row, new_col, cell_pos) =
                                proportion * (current_height - neighbor_height);
                        }
                    }
                }
            }
        }
    }
}

__global__ void compute_spillage_propagation_kernel(int rows, int columns, int *d_water_level, float *d_spillage_flag,
                                                   float *d_spillage_level, float *d_spillage_from_neigh,
                                                   int *d_max_spillage_iter) {
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_pos >= rows || col_pos >= columns) return;

    int depths = CONTIGUOUS_CELLS;
    int cell_pos;

    if (accessMat(d_spillage_flag, row_pos, col_pos) == 1) {
        accessMat(d_water_level, row_pos, col_pos) -=
            FIXED(accessMat(d_spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR);
    }

    // Compute termination condition: Maximum cell spillage during the iteration
    // Note: atomic max works only for floats 
    int current_spill = FIXED(accessMat(d_spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR);
    atomicMax(d_max_spillage_iter, current_spill); 

    for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        accessMat(d_water_level, row_pos, col_pos) +=
            FIXED(accessMat3D(d_spillage_from_neigh, row_pos, col_pos, cell_pos) / SPILLAGE_FACTOR);
    }
}

__global__ void compute_private_spillage_propagation_kernel(
    int rows, int columns,
    int *d_water_level,
    float *d_spillage_flag,
    float *d_spillage_level,
    float *d_spillage_from_neigh,
    int *d_max_spillage_iter)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // Thread ID within the block
    int row_stride = blockDim.y * gridDim.y;
    int col_stride = blockDim.x * gridDim.x;


    for (int r = row; r < rows; r += row_stride) {
        for (int c = col; c < columns; c += col_stride) {
            extern __shared__ int smax[]; // Shared memory for maximum spillage in the iteration (fixed-point representation)
            int local_max = 0; // Local variable to track the maximum spillage for this thread (fixed-point representation)


            if (r < rows && c < columns) { // Check if the thread is within bounds (Can't return early do to syncthreads)
                if (accessMat(d_spillage_flag, r, c) == 1.0f) {
                    // Eliminate the spillage from the origin cell
                    int out_fixed = FIXED(accessMat(d_spillage_level, r, c) / SPILLAGE_FACTOR);
                    accessMat(d_water_level, r, c) -= out_fixed; // Update water level with fixed-point spillage
                    local_max = out_fixed; // Update local maximum spillage for this thread
                }

                // Accumulate spillage from neighbors
                for (int cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
                    int depths = CONTIGUOUS_CELLS;
                    accessMat(d_water_level, r, c) +=
                        FIXED(accessMat3D(d_spillage_from_neigh, r, c, cell_pos) / SPILLAGE_FACTOR);
                }
            }

            smax[tid] = local_max; // Store the local maximum spillage for this thread in shared memory
            __syncthreads(); // Ensure all threads have written their local maximum to shared memory before reduction

            // Perform parallel reduction to find the maximum spillage in the iteration
            // NOTE: Requires blockDim.x * blockDim.y to be a power of 2 for this reduction to work correctly
            for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride /= 2) { // >>= 1?
                if (tid < stride && smax[tid + stride] > smax[tid]) {
                    smax[tid] = smax[tid + stride];
                }
                __syncthreads();
            }

            if (tid == 0) atomicMax(d_max_spillage_iter, smax[0]);
        }
    }
}

__global__ void reset_ancillary_structures_kernel(int rows, int columns, float *d_spillage_flag,
                                                   float *d_spillage_level, float *d_spillage_from_neigh) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int cell_pos;

    if (row >= rows || col >= columns)
        return;

    for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        int depths = CONTIGUOUS_CELLS;
        accessMat3D(d_spillage_from_neigh, row, col, cell_pos) = 0.0f;
    }
    accessMat(d_spillage_flag, row, col) = 0.0f;
    accessMat(d_spillage_level, row, col) = 0.0f;
}

/*
 * Main compute function
 */
extern "C" void do_compute(struct parameters *p, struct results *r) {
    int rows = p->rows, columns = p->columns;
    printf("Starting CUDA version with %d rows and %d columns\n", rows, columns);
    int *minute = &r->minute;

    size_t cell_count = (size_t)rows * (size_t)columns;
    size_t neigh_count = cell_count * (size_t)CONTIGUOUS_CELLS;

    /* 2. Start global timer */
    CUDA_CHECK_FUNCTION(cudaSetDevice(0));
    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

    /*
     *
     * Allocate memory and call kernels in this function.
     * Ensure all debug and animation code works in your final version.
     *
     */

    /* Memory allocation */

    int *h_water_level;           // Level of water on each cell (fixed precision)
    float *h_ground;              // Ground height
    unsigned long long h_total_water_loss; // Device-side accumulator copy (atomicAdd-compatible)
    unsigned long long h_total_rainfall; // Device-side accumulator for total water loss
    int h_max_spillage_iter; // Fixed-point max spillage in the current iteration
    int h_max_spillage_scenario; // Fixed-point max spillage during the scenario
    int h_max_spillage_minute; // Minute of the maximum spillage during the scenario (for statistics)
    float *h_spillage_flag;       // Indicates which cells are spilling to neighbors
    float *h_spillage_level;      // Maximum level of spillage of each cell
    float *h_spillage_from_neigh; // Spillage from each neighbor

    unsigned long long *d_total_water_loss = NULL;
    unsigned long long *d_total_rainfall = NULL;
    int *d_max_spillage_iter = NULL;
    int *d_water_level = NULL;

    Cloud_soa_t d_clouds_soa = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    float *d_ground = NULL;

    float *d_spillage_flag = NULL;
    float *d_spillage_level = NULL;
    float *d_spillage_from_neigh = NULL;

    h_ground = p->ground;
    h_total_water_loss = 0;
    h_total_rainfall = 0;
    h_max_spillage_iter = INT_MAX;
    h_max_spillage_scenario = 0;
    h_max_spillage_minute = 0;
    h_water_level = (int *)malloc(sizeof(int) * cell_count);
    h_spillage_flag = (float *)malloc(sizeof(float) * cell_count);
    h_spillage_level = (float *)malloc(sizeof(float) * cell_count);
    h_spillage_from_neigh = (float *)malloc(sizeof(float) * neigh_count);

    if (h_water_level == NULL || h_spillage_flag == NULL || h_spillage_level == NULL || h_spillage_from_neigh == NULL) {
        fprintf(stderr, "-- Error allocating ground and rain structures for size: %d x %d \n", rows, columns);
        exit(EXIT_FAILURE);
    }


    /*
     * Allocate memory on the GPU and copy the initial ground heights
     */
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_ground, sizeof(float) * cell_count));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_total_water_loss, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_total_rainfall, sizeof(unsigned long long)));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_max_spillage_iter, sizeof(int)));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_clouds_soa.x, sizeof(float) * (size_t)p->num_clouds));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_clouds_soa.y, sizeof(float) * (size_t)p->num_clouds));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_clouds_soa.radius, sizeof(float) * (size_t)p->num_clouds));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_clouds_soa.intensity, sizeof(float) * (size_t)p->num_clouds));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_clouds_soa.sqrt_divr_intensity, sizeof(float) * (size_t)p->num_clouds));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_clouds_soa.speed, sizeof(float) * (size_t)p->num_clouds));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_clouds_soa.angle, sizeof(float) * (size_t)p->num_clouds));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_clouds_soa.active, sizeof(int) * (size_t)p->num_clouds));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_water_level, sizeof(int) * cell_count));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_spillage_flag, sizeof(float) * cell_count));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_spillage_level, sizeof(float) * cell_count));
    CUDA_CHECK_FUNCTION(cudaMalloc((void **)&d_spillage_from_neigh, sizeof(float) * neigh_count));


    CUDA_CHECK_FUNCTION(cudaMemcpy(d_ground, h_ground, sizeof(float) * cell_count, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds_soa.x, p->clouds_soa.x, sizeof(float) * (size_t)p->num_clouds, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds_soa.y, p->clouds_soa.y, sizeof(float) * (size_t)p->num_clouds, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds_soa.radius, p->clouds_soa.radius, sizeof(float) * (size_t)p->num_clouds, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds_soa.intensity, p->clouds_soa.intensity, sizeof(float) * (size_t)p->num_clouds, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds_soa.sqrt_divr_intensity, p->clouds_soa.sqrt_divr_intensity, sizeof(float) * (size_t)p->num_clouds, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds_soa.speed, p->clouds_soa.speed, sizeof(float) * (size_t)p->num_clouds, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds_soa.angle, p->clouds_soa.angle, sizeof(float) * (size_t)p->num_clouds, cudaMemcpyHostToDevice));
    CUDA_CHECK_FUNCTION(cudaMemcpy(d_clouds_soa.active, p->clouds_soa.active, sizeof(int) * (size_t)p->num_clouds, cudaMemcpyHostToDevice));


    dim3 block(32, 8); // Adjust for GPU architecture and problem size
    dim3 grid((columns + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    init_state_kernel<<<grid, block>>>(rows, columns, d_water_level, d_spillage_flag, d_spillage_level,
                                       d_spillage_from_neigh);
    CUDA_CHECK_KERNEL();
    //CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

#ifdef DEBUG
    print_matrix(PRECISION_FLOAT, rows, columns, h_ground, "Ground heights");
#ifndef ANIMATION
    print_clouds(p->num_clouds, p->clouds);
#endif
#endif

    /* Prepare to measure runtime */
    r->runtime = get_time();

    /* Flood simulation */
    for (*minute = 0; *minute < p->num_minutes && FLOATING(h_max_spillage_iter) > p->threshold; (*minute)++) {

        CUDA_CHECK_FUNCTION(cudaMemset(d_total_rainfall, 0, sizeof(unsigned long long)));
        CUDA_CHECK_FUNCTION(cudaMemset(d_total_water_loss, 0, sizeof(unsigned long long)));

        /* Step 1: Cloud movement and rainfall */
        /* Step 1.1: Cloud movement */
        cloud_movement_kernel_soa<<<(p->num_clouds + block.x - 1) / block.x, block.x>>>(
            p->num_clouds, d_clouds_soa.x, d_clouds_soa.y, d_clouds_soa.speed, d_clouds_soa.angle,
            d_clouds_soa.active);
        CUDA_CHECK_KERNEL();
        //CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());

#ifdef DEBUG
#ifndef ANIMATION
        CUDA_CHECK_FUNCTION(
            cudaMemcpy(p->clouds_soa.x, d_clouds_soa.x, sizeof(float) * (size_t)p->num_clouds, cudaMemcpyDeviceToHost));
        CUDA_CHECK_FUNCTION(
            cudaMemcpy(p->clouds_soa.y, d_clouds_soa.y, sizeof(float) * (size_t)p->num_clouds, cudaMemcpyDeviceToHost));
        for (int cloud = 0; cloud < p->num_clouds; cloud++) {
            p->clouds[cloud].x = p->clouds_soa.x[cloud];
            p->clouds[cloud].y = p->clouds_soa.y[cloud];
        }
        print_clouds(p->num_clouds, p->clouds);
#endif
#endif

        /* Step 1.2: Rainfall */
        size_t rainfall_shared_mem = (size_t)(block.x * block.y) * (5 * sizeof(float) + sizeof(int));
        alt_calc_rainfall_kernel_soa<<<grid, block, rainfall_shared_mem>>>(
            rows, columns, p->num_clouds, d_total_rainfall, d_clouds_soa.x, d_clouds_soa.y,
            d_clouds_soa.radius, d_clouds_soa.intensity, d_clouds_soa.sqrt_divr_intensity,
            d_clouds_soa.active, p->ex_factor, d_water_level);
        //alternative_rainfall_kernel<<<grid, block>>>(rows, columns, p->num_clouds, d_total_rainfall, d_clouds, p->ex_factor, d_water_level);
        //rainfall_kernel<<<(p->num_clouds + block.x - 1) / block.x, block.x>>>(p->num_clouds, d_total_rainfall, d_clouds, rows, columns, p->ex_factor, d_water_level);
        CUDA_CHECK_KERNEL();
        //CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());


#ifdef DEBUG
    print_matrix(PRECISION_FIXED, rows, columns, h_water_level, "Water after rain");
#endif
        /* Step 2: Compute candidate spillage to neighbors. */
        compute_spillage_kernel<<<grid, block>>>(rows, columns, d_total_water_loss, d_ground, d_water_level, d_spillage_flag,
                                                 d_spillage_level, d_spillage_from_neigh);

        CUDA_CHECK_KERNEL();


        /* Step 3: Propagation of previously computer water spillage to/from neighbors */
        CUDA_CHECK_FUNCTION(cudaMemset(d_max_spillage_iter, 0, sizeof(int)));

        // compute_spillage_propagation_kernel<<<grid, block>>>(rows, columns, d_water_level, d_spillage_flag, d_spillage_level, d_spillage_from_neigh,
        //                             d_max_spillage_iter);
        
        compute_private_spillage_propagation_kernel<<<grid, block, block.x * block.y * sizeof(int)>>>(rows, columns, d_water_level, d_spillage_flag, d_spillage_level, d_spillage_from_neigh,
                                                    d_max_spillage_iter);
        CUDA_CHECK_KERNEL();


#ifdef DEBUG
#ifndef ANIMATION
    print_matrix(PRECISION_FIXED, rows, columns, h_water_level, "Water after spillage");
#endif
#endif

        /* Reset ancillary structures */
        reset_ancillary_structures_kernel<<<grid, block>>>(rows, columns, d_spillage_flag, d_spillage_level,
                                                           d_spillage_from_neigh);
        CUDA_CHECK_KERNEL();


        //CUDA_CHECK_FUNCTION(cudaMemcpy(h_water_level, d_water_level, sizeof(int) * cell_count, cudaMemcpyDeviceToHost));
        CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_rainfall, d_total_rainfall, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        r->total_rain += (long)h_total_rainfall;

        CUDA_CHECK_FUNCTION(cudaMemcpy(&h_total_water_loss, d_total_water_loss, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        r->total_water_loss += (long)h_total_water_loss;

        CUDA_CHECK_FUNCTION(cudaMemcpy(&h_max_spillage_iter, d_max_spillage_iter, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_max_spillage_iter > h_max_spillage_scenario) {
            h_max_spillage_scenario = h_max_spillage_iter;
            h_max_spillage_minute = *minute;
        }
    }

    cudaDeviceSynchronize();

    r->runtime = get_time() - r->runtime;

    r->max_spillage_scenario = FLOATING(h_max_spillage_scenario);
    r->max_spillage_minute = h_max_spillage_minute;

    CUDA_CHECK_FUNCTION(cudaMemcpy(h_water_level, d_water_level, sizeof(int) * cell_count, cudaMemcpyDeviceToHost));

    if (p->final_matrix) {
        print_matrix(PRECISION_FIXED, rows, columns, h_water_level, "Water after spillage");
    }

    /* Statistics: Total remaining water and maximum amount of water in a cell */
    r->max_water_scenario = 0.0;
    for (int row_pos = 0; row_pos < rows; row_pos++) {
        for (int col_pos = 0; col_pos < columns; col_pos++) {
            if (FLOATING(accessMat(h_water_level, row_pos, col_pos)) > r->max_water_scenario)
                r->max_water_scenario = FLOATING(accessMat(h_water_level, row_pos, col_pos));
            r->total_water += accessMat(h_water_level, row_pos, col_pos);
        }
    }

    /* Free resources */
    free(h_ground);
    free(h_water_level);
    free(h_spillage_flag);
    free(h_spillage_level);
    free(h_spillage_from_neigh);

    CUDA_CHECK_FUNCTION(cudaFree(d_ground));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_water_loss));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds_soa.x));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds_soa.y));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds_soa.radius));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds_soa.intensity));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds_soa.sqrt_divr_intensity));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds_soa.speed));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds_soa.angle));
    CUDA_CHECK_FUNCTION(cudaFree(d_clouds_soa.active));
    CUDA_CHECK_FUNCTION(cudaFree(d_water_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_total_rainfall));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_flag));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_level));
    CUDA_CHECK_FUNCTION(cudaFree(d_spillage_from_neigh));
    CUDA_CHECK_FUNCTION(cudaFree(d_max_spillage_iter));


    CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
}
