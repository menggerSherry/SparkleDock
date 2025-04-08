#include "initialization.h"
#include "lib_tools.h"
void init_center_translate(
    double * center_coord, double * coord, 
    vector<string> element, int * structure_size, 
    const int num_atom, const int num_structure){
    // TODO: multi atoms
    int num_threads = get_env_num_threads();
    int start_strutures = structure_size[0];
    int end_structures = structure_size[1];

    double total_x = 0.0;
    double total_y = 0.0;
    double total_z = 0.0;
    int dimension = 0;
    #pragma omp parallel for num_threads(num_threads)
    for(int i = start_strutures; i< end_structures; i++){
        if(element[i] != "H"){
            total_x += coord[i*3];
            total_y += coord[i*3+1];
            total_z += coord[i*3+2];
            dimension ++;
        }
    }

    center_coord[0] = -1 * total_x / dimension;
    center_coord[1] = -1 * total_y / dimension;
    center_coord[2] = -1 * total_z / dimension;
}


