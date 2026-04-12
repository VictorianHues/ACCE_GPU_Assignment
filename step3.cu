__global__ void step3_kernel(){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col blockIdx.x * blockDim.x + threadIdx.x;

    // add a check for rows and columns number
    // rows and column position substitute 
    // If the cell has spillage
    if (accessMat(spillage_flag, row_pos, col_pos) == 1) {

        // Eliminate the spillage from the origin cell
        accessMat(water_level, row_pos, col_pos) -=
            FIXED(accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR);

        
        // Compute termination condition: Maximum cell spillage during the iteration
        // Not safe for the kernel, needs to be deleted 
        if (accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR > max_spillage_iter) {
            max_spillage_iter = accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR;
        }
        // Statistics: Record maximum cell spillage during the scenario and its time
        // Not safe for the kernel, needs to be deleted
        if (accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR > r->max_spillage_scenario) {
            r->max_spillage_scenario = accessMat(spillage_level, row_pos, col_pos) / SPILLAGE_FACTOR;
            r->max_spillage_minute = *minute;
        }
    }

    // Accumulate spillage from neighbors
    for (cell_pos = 0; cell_pos < CONTIGUOUS_CELLS; cell_pos++) {
        int depths = CONTIGUOUS_CELLS;
        accessMat(water_level, row_pos, col_pos) +=
            FIXED(accessMat3D(spillage_from_neigh, row_pos, col_pos, cell_pos) / SPILLAGE_FACTOR);
    }
}