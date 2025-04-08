#include "translate.h"
#include "lib_tools.h"

void translate(
    double * coord, double * translate_center_coo, int * structure_size, 
    const int num_atom, const int num_structure
){
    int num_threads = get_env_num_threads();
    int start_strutures = structure_size[0];
    int end_structures = structure_size[1];
    #pragma omp parallel for num_threads(num_threads)
    for(int i = start_strutures;i<end_structures;i++){
        coord[i*3] += translate_center_coo[0];
        coord[i*3+1] += translate_center_coo[1];
        coord[i*3+2] += translate_center_coo[2];
    }
}