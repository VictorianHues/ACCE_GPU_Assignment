__global__ void step3_kernel(
    int *water_level,
    float *spillage_flag,
    float *spillage_level,
    float *spillage_from_neigh,
    int rows,
    int columns
) {
    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_pos >= rows || col_pos >= columns) return;

    int depths = CONTIGUOUS_CELLS; // needed?
    int cell_pos;

    if (accessMat(spillage_flag, row_pos, col_pos) == 1) {
        accessMat(water_level, row_pos, col_pos) -=
            FIXED(accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR);
    }

    for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        accessMat(water_level, row_pos, col_pos) +=
            FIXED(accessMat3D(spillage_from_neigh, row_pos, col_pos, cell_pos) / SPILLAGE_FACTOR);
    }
}