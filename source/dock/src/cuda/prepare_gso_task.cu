#include "prepare_gso_tasks.h"
#include "swarm_centers.h"
#include "fastdfire.h"
#include <iostream>
#include <random>
#include <cstdint>
#include "lib_tools.h"
#include <cmath>
#include <string.h>
#include <fstream>
#include "interface.h"
#include <sstream>
#include <iomanip>  // 包含设置小数点精度的头文件
#include "complex.h"
#include <algorithm>
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "error.cuh"
#include "path_include.h"
#include <mma.h>
#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdint.h>
using namespace cooperative_groups;
// using namespace nvcuda;
#define MAX_TRANSLATION 30
#define MAX_ROTATION 1.0
#define MIN_EXTENT 0.0
#define MAX_EXTENT 1.0
#define DEFAULT_NMODES_REC 10
#define DEFAULT_NMODES_LIG 10

#define STEPS 100
const unsigned int FULL_MASK=0xffffffff;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


__forceinline__ __device__ 
void rotate(
    double * coord, double * rotate, 
    double * rotate_inverse, double * current_pos, double * current_pos2,
    int atom_num
)
{
    // rotation
                
    double weight = rotate[0];
    double pos_x = rotate[1];
    double pos_y = rotate[2];
    double pos_z = rotate[3];

    

    double rotate_norm2 = weight * weight + pos_x * pos_x + pos_y * pos_y + pos_z * pos_z;
    rotate_inverse[0] = rotate[0] / rotate_norm2;
    rotate_inverse[1] = - rotate[1] / rotate_norm2;
    rotate_inverse[2] = - rotate[2] / rotate_norm2;
    rotate_inverse[3] = - rotate[3] / rotate_norm2;
    
    #pragma unroll
    for(int i = 0; i<atom_num; i++){
        current_pos[0] = 0;
        // double w = 0 ;
        current_pos[1] = coord[i*3];
        current_pos[2] = coord[i*3+1];
        current_pos[3] = coord[i*3+2];
        // memcpy(current_pos+1, &coord[i*3], sizeof(double)*3);

        // qmul(current_pos2,rotate,current_pos);
        current_pos2[0] = (rotate[0] * current_pos[0] - rotate[1] * current_pos[1] - rotate[2] * current_pos[2] - rotate[3] * current_pos[3]);
        current_pos2[1] = (rotate[0] * current_pos[1] + rotate[1] * current_pos[0] + rotate[2] * current_pos[3] - rotate[3] * current_pos[2]);
        current_pos2[2] = (rotate[0] * current_pos[2] - rotate[1] * current_pos[3] + rotate[2] * current_pos[0] + rotate[3] * current_pos[1]);
        current_pos2[3] = (rotate[0] * current_pos[3] + rotate[1] * current_pos[2] - rotate[2] * current_pos[1] + rotate[3] * current_pos[0]);
        

        // qmul(current_pos, current_pos2, rotate_inverse);
        current_pos[0] = (current_pos2[0] * rotate_inverse[0] - current_pos2[1] * rotate_inverse[1] - current_pos2[2] * rotate_inverse[2] - current_pos2[3] * rotate_inverse[3]);
        current_pos[1] = (current_pos2[0] * rotate_inverse[1] + current_pos2[1] * rotate_inverse[0] + current_pos2[2] * rotate_inverse[3] - current_pos2[3] * rotate_inverse[2]);
        current_pos[2] = (current_pos2[0] * rotate_inverse[2] - current_pos2[1] * rotate_inverse[3] + current_pos2[2] * rotate_inverse[0] + current_pos2[3] * rotate_inverse[1]);
        current_pos[3] = (current_pos2[0] * rotate_inverse[3] + current_pos2[1] * rotate_inverse[2] - current_pos2[2] * rotate_inverse[1] + current_pos2[3] * rotate_inverse[0]);
        // std::cout<<current_pos[0]<<" "<<current_pos[1]<<" "<<current_pos[2]<<" "<<current_pos[3]<<std::endl;
        coord[i*3] = current_pos[1];
        coord[i*3+1] = current_pos[2];
        coord[i*3+2] = current_pos[3];

        // memcpy(&coord[i*3], &current_pos[1], sizeof(double)*3);
        
    }


}


__device__ __forceinline__
void compute_probability_moving_from_neighbors_gpu(
    double * probabilities, int * neighbors, double * luciferin, int nnei_len, double current_luciferin
){
    /*
    该方法目前存疑
    probabilities: 概率
    neighbors： 对应swarm ， 对应glowwormd的 邻居表
    luciferin： 当前swarm的luciferin
    nnei_len：当前 swarm 当前glowworm的邻居表长度
    current_luciferin：当前原子的luciferin
    */
    double total_sum = 0;
    // double * probabilities = new double [current_glowworm->nnei_len];
    for(int idx = 0; idx<nnei_len; idx++){
        int nei_id = neighbors[idx];
        // double current_luciferin = luciferin[idx];
        double neighbor_luciferin = luciferin[nei_id];
        double difference = neighbor_luciferin - current_luciferin;
        probabilities[idx] = difference;
        total_sum += difference;

    }
    for(int idx = 0; idx<nnei_len; idx++){
        probabilities[idx] /= total_sum;
    }
}


__global__ void prepare_receptor(
    double * d_receptor_pose, double * d_receptor_atom_coordinates, double * d_centers_pos, double * d_receptor_modes,
    int * d_receptor_mask,
    int swarms, int num_glowworms, int anm_lig, int anm_rec, int receptor_atoms, int rec_pad_atoms
){
    const int block_idx = blockIdx.x;
    const int block_idy = blockIdx.y;
    const int block_idz = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;
    __shared__ double s_current_pose[1024];
    int pos_len = 7+anm_rec+anm_lig;
    double * pos_gloworm = &d_centers_pos[block_idz*num_glowworms*pos_len + block_idy*pos_len];
    double * current_receptor_pose = &d_receptor_pose[block_idz * num_glowworms * rec_pad_atoms * 4 + block_idy * rec_pad_atoms * 4];
    int n = block_size * block_idx + thread_idx;

    s_current_pose[thread_idx*3] = (n < receptor_atoms) ?  d_receptor_atom_coordinates[n*3] : 0;
    s_current_pose[thread_idx*3+1] = (n < receptor_atoms) ? d_receptor_atom_coordinates[n*3+1] : 0;
    s_current_pose[thread_idx*3+2] = (n < receptor_atoms) ? d_receptor_atom_coordinates[n*3+2] : 0;

    __syncthreads();

    if(n<receptor_atoms){
        for(int j = 0; j< DEFAULT_NMODES_REC; j++){
            double * rep_modes = &d_receptor_modes[j*receptor_atoms*3+n*3];
            double rec_extent = pos_gloworm[7+j];
            int mask = d_receptor_mask[n];
            s_current_pose[thread_idx*3] += rep_modes[0] * rec_extent * mask;
            s_current_pose[thread_idx*3+1] += rep_modes[1] * rec_extent * mask;
            s_current_pose[thread_idx*3+2] += rep_modes[2] * rec_extent * mask;
        }
        current_receptor_pose[n*4] = s_current_pose[thread_idx*3];
        current_receptor_pose[n*4+1] = s_current_pose[thread_idx*3+1];
        current_receptor_pose[n*4+2] = s_current_pose[thread_idx*3+2];
        current_receptor_pose[n*4+3] = 0;

    }
    
}
__global__ void prepare_ligand(
    double * d_ligand_pose, double * d_ligand_atom_coordinates, double * d_current_pos, double * d_ligand_modes,
    int * d_ligand_mask,
    int swarms, int num_glowworms, int anm_lig, int anm_rec, int ligand_atoms, int lig_pad_atoms
){
    const int block_idx = blockIdx.x;
    const int block_idy = blockIdx.y;
    const int block_idz = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;
    __shared__ double s_current_pose[1024];
    int pos_len = 7+anm_rec+anm_lig;
    double * pos_glowworm = &d_current_pos[block_idz*num_glowworms*pos_len + block_idy*pos_len];
    double * current_ligand_pose = &d_ligand_pose[block_idz * num_glowworms * lig_pad_atoms * 4 + block_idy * lig_pad_atoms * 4];
    int n = block_size * block_idx + thread_idx;
    s_current_pose[thread_idx*3] = d_ligand_atom_coordinates[n*3];
    s_current_pose[thread_idx*3+1] = d_ligand_atom_coordinates[n*3+1];
    s_current_pose[thread_idx*3+2] = d_ligand_atom_coordinates[n*3+2];
    __syncthreads();


    if(n<ligand_atoms){
        for(int j = 0; j< DEFAULT_NMODES_LIG; j++){
            double * lig_modes = &d_ligand_modes[j*ligand_atoms*3 + n*3];
            double lig_extent = pos_glowworm[7+DEFAULT_NMODES_REC+j];
            int mask = d_ligand_mask[n];
            s_current_pose[thread_idx*3] += lig_modes[0] * lig_extent * mask;
            s_current_pose[thread_idx*3+1] += lig_modes[1] * lig_extent * mask;
            s_current_pose[thread_idx*3+2] += lig_modes[2] * lig_extent * mask;
        }

        double * rotate = &pos_glowworm[3];
    
    
        double weight = rotate[0];
        double pos_x = rotate[1];
        double pos_y = rotate[2];
        double pos_z = rotate[3];

        

        double rotate_norm2 = weight * weight + pos_x * pos_x + pos_y * pos_y + pos_z * pos_z;
        double rotate_inverse_0 = rotate[0] / rotate_norm2;
        double rotate_inverse_1 = - rotate[1] / rotate_norm2;
        double rotate_inverse_2 = - rotate[2] / rotate_norm2;
        double rotate_inverse_3 = - rotate[3] / rotate_norm2;
        
        
        
        
        double current_pos_0 = 0;
        // double w = 0 ;
        double current_pos_1 = s_current_pose[thread_idx*3];
        double current_pos_2 = s_current_pose[thread_idx*3+1];
        double current_pos_3 = s_current_pose[thread_idx*3+2];
        // memcpy(current_pos+1, &coord[i*3], sizeof(double)*3);

        // qmul(current_pos2,rotate,current_pos);
        double current_pos2_0 = (rotate[0] * current_pos_0 - rotate[1] * current_pos_1 - rotate[2] * current_pos_2 - rotate[3] * current_pos_3);
        double current_pos2_1 = (rotate[0] * current_pos_1 + rotate[1] * current_pos_0 + rotate[2] * current_pos_3 - rotate[3] * current_pos_2);
        double current_pos2_2 = (rotate[0] * current_pos_2 - rotate[1] * current_pos_3 + rotate[2] * current_pos_0 + rotate[3] * current_pos_1);
        double current_pos2_3 = (rotate[0] * current_pos_3 + rotate[1] * current_pos_2 - rotate[2] * current_pos_1 + rotate[3] * current_pos_0);
        

        // qmul(current_pos, current_pos2, rotate_inverse);
        current_pos_0 = (current_pos2_0 * rotate_inverse_0 - current_pos2_1 * rotate_inverse_1 - current_pos2_2 * rotate_inverse_2 - current_pos2_3 * rotate_inverse_3);
        current_pos_1 = (current_pos2_0 * rotate_inverse_1 + current_pos2_1 * rotate_inverse_0 + current_pos2_2 * rotate_inverse_3 - current_pos2_3 * rotate_inverse_2);
        current_pos_2 = (current_pos2_0 * rotate_inverse_2 - current_pos2_1 * rotate_inverse_3 + current_pos2_2 * rotate_inverse_0 + current_pos2_3 * rotate_inverse_1);
        current_pos_3 = (current_pos2_0 * rotate_inverse_3 + current_pos2_1 * rotate_inverse_2 - current_pos2_2 * rotate_inverse_1 + current_pos2_3 * rotate_inverse_0);
        // std::cout<<current_pos_0<<" "<<current_pos_1<<" "<<current_pos_2<<" "<<current_pos_3<<std::endl;
        s_current_pose[thread_idx*3] = current_pos_1;
        s_current_pose[thread_idx*3+1] = current_pos_2;
        s_current_pose[thread_idx*3+2] = current_pos_3;

        s_current_pose[thread_idx*3] += pos_glowworm[0];
        s_current_pose[thread_idx*3+1] +=  pos_glowworm[1];
        s_current_pose[thread_idx*3+2] +=  pos_glowworm[2];

        current_ligand_pose[n*4] = s_current_pose[thread_idx*3];
        current_ligand_pose[n*4+1] = s_current_pose[thread_idx*3+1];
        current_ligand_pose[n*4+2] = s_current_pose[thread_idx*3+2];
        current_ligand_pose[n*4+3] = 0;

    }
}

__global__ void prepare_receptor_pose(
    double * d_centers_pos, double * d_receptor_pose, 
    double * d_receptor_atom_coordinates, 
    double * d_receptor_modes, 
    int * d_receptor_mask, 
    int * d_moved_base, int * d_step_base,
    size_t swarms, size_t num_glowworms, size_t anm_lig, size_t anm_rec,
    size_t receptor_atoms, size_t rec_pad_atoms
){
    /*
    rotate 内存创建
    */
    const size_t block_idx = blockIdx.x;
    const size_t block_idy = blockIdx.y;
    const size_t block_idz = blockIdx.z;
    const size_t thread_idx = threadIdx.x;
    const size_t block_size = blockDim.x;
    __shared__ int s_mem[8192];
    double * s_current_pose = (double *)s_mem;
    int * s_mask = s_mem+2048;
    // __shared__ double s_current_pose[1024];
    int pos_len = 7+anm_rec+anm_lig;

    double * pos_gloworm = &d_centers_pos[block_idz*num_glowworms*pos_len + block_idy*pos_len];
    double * current_receptor_pose = &d_receptor_pose[block_idz * num_glowworms * rec_pad_atoms * 4 + block_idy * rec_pad_atoms*4];
    // printf("test\n");
    size_t n = block_size * block_idx + thread_idx;
    // double * s_pos = s_mem;
    // double * s_current_pose = s_mem;

    s_current_pose[thread_idx*3] = d_receptor_atom_coordinates[n*3];
    s_current_pose[thread_idx*3+1] = d_receptor_atom_coordinates[n*3+1];
    s_current_pose[thread_idx*3+2] = d_receptor_atom_coordinates[n*3+2];
    s_mask[thread_idx] = d_receptor_mask[n];
    __syncthreads();

    double center_pos [27];
    for(int i = 0; i<27; i++){
        center_pos[i] = pos_gloworm[i];
    }
    int moved = d_moved_base[block_idz*num_glowworms + block_idy];
    int steped = d_step_base[block_idz*num_glowworms + block_idy];
    
    if (n<receptor_atoms && (moved || steped == 0)){
        for(int jj = 0; jj< 3;jj++){
            current_receptor_pose[n*3 +jj] = d_receptor_atom_coordinates[n*3+jj];
        }
        for(int j = 0; j<DEFAULT_NMODES_REC; j++){
            double * rep_modes = &d_receptor_modes[j*receptor_atoms*3+n*3];
            double rec_instant = center_pos[7+j]; 
            int mask = s_mask[thread_idx];
            s_current_pose[thread_idx*3] += rep_modes[0] * rec_instant * mask;
            s_current_pose[thread_idx*3+1] += rep_modes[1] * rec_instant * mask;
            s_current_pose[thread_idx*3+2] += rep_modes[2] * rec_instant * mask;
        }
        
        current_receptor_pose[n*4] = s_current_pose[thread_idx*3];
        current_receptor_pose[n*4+1] = s_current_pose[thread_idx*3+1];
        current_receptor_pose[n*4+2] = s_current_pose[thread_idx*3+2];
        current_receptor_pose[n*4+3] = 0;
        // if(block_idz==3 && block_idy==3 && block_idx==0 && thread_idx==0){
        //     printf("%f %f %f \n", current_receptor_pose[n*3],current_receptor_pose[n*3+1],current_receptor_pose[n*3+2]);
        // }
    }

}

__global__ void prepare_lignad_pose(
    double * d_centers_pos,double * d_ligand_pose, double * d_ligand_reference_pose,
    double * d_ligand_atom_coordinates,  
    double * d_current_ligand_reference_pose, 
    double * d_ligand_modes,
    int * d_ligand_mask,
    int * d_moved_base, int * d_step_base,
    size_t swarms, size_t num_glowworms, size_t anm_lig, size_t anm_rec,
    size_t ligand_atoms, size_t lig_pos
){
    const size_t block_idx = blockIdx.x;
    const size_t block_idy = blockIdx.y;
    const size_t block_idz = blockIdx.z;
    const size_t thread_idx = threadIdx.x;
    const size_t block_size = blockDim.x;
    double * pos_gloworm = &d_centers_pos[block_idz*num_glowworms*(7+anm_rec+anm_lig) + block_idy*(7+anm_rec+anm_lig)];
    // share 
    double * current_ligand_pose = &d_ligand_pose[block_idz * num_glowworms * lig_pos * 4 + block_idy * lig_pos * 4];
    double * current_ligand_reference_pose = &d_ligand_reference_pose[block_idz * num_glowworms * 3 + block_idy * 3];
    __shared__ int s_mem[8192];
    double * s_current_pose = (double *)s_mem;
    int * s_mask = s_mem+2048;
    
    // __shared__ double s_current_pose[1024];

    int moved = d_moved_base[block_idz*num_glowworms + block_idy];
    int steped = d_step_base[block_idz*num_glowworms + block_idy];
    size_t n = block_size * block_idx + thread_idx;

    s_current_pose[thread_idx*3] = d_ligand_atom_coordinates[n*3];
    s_current_pose[thread_idx*3+1] = d_ligand_atom_coordinates[n*3+1];
    s_current_pose[thread_idx*3+2] = d_ligand_atom_coordinates[n*3+2];

    s_mask[thread_idx] = d_ligand_mask[n];
    __syncthreads();

    double center_pos [27];
    for(int i = 0; i<27; i++){
        center_pos[i] = pos_gloworm[i];
    }
    if(n < ligand_atoms && (moved || steped == 0)){
        // for(int jj = 0; jj<3;jj++){
        //     current_ligand_pose[n*3+jj] = d_ligand_atom_coordinates[n*3+jj];
        // }
        #pragma unroll
        for(int j = 0;j<DEFAULT_NMODES_LIG; j++){

            double * rep_modes = &d_ligand_modes[j*ligand_atoms*3+n*3]; //
            double rec_instant = center_pos[7+DEFAULT_NMODES_REC+j];
            int mask  = s_mask[thread_idx];
            // if(mask == 1){
            s_current_pose[thread_idx*3] += rep_modes[0] * rec_instant * mask;
            s_current_pose[thread_idx*3+1] += rep_modes[1] * rec_instant * mask;
            s_current_pose[thread_idx*3+2] += rep_modes[2] * rec_instant * mask;
                // }
            

        }

        double * rotate = &center_pos[3];
    
    
        double weight = rotate[0];
        double pos_x = rotate[1];
        double pos_y = rotate[2];
        double pos_z = rotate[3];

        

        double rotate_norm2 = weight * weight + pos_x * pos_x + pos_y * pos_y + pos_z * pos_z;
        double rotate_inverse_0 = rotate[0] / rotate_norm2;
        double rotate_inverse_1 = - rotate[1] / rotate_norm2;
        double rotate_inverse_2 = - rotate[2] / rotate_norm2;
        double rotate_inverse_3 = - rotate[3] / rotate_norm2;
        
        
        
        
        double current_pos_0 = 0;
        // double w = 0 ;
        double current_pos_1 = s_current_pose[thread_idx*3];
        double current_pos_2 = s_current_pose[thread_idx*3+1];
        double current_pos_3 = s_current_pose[thread_idx*3+2];
        // memcpy(current_pos+1, &coord[i*3], sizeof(double)*3);

        // qmul(current_pos2,rotate,current_pos);
        double  current_pos2_0 = (rotate[0] * current_pos_0 - rotate[1] * current_pos_1 - rotate[2] * current_pos_2 - rotate[3] * current_pos_3);
        double current_pos2_1 = (rotate[0] * current_pos_1 + rotate[1] * current_pos_0 + rotate[2] * current_pos_3 - rotate[3] * current_pos_2);
        double current_pos2_2 = (rotate[0] * current_pos_2 - rotate[1] * current_pos_3 + rotate[2] * current_pos_0 + rotate[3] * current_pos_1);
        double current_pos2_3 = (rotate[0] * current_pos_3 + rotate[1] * current_pos_2 - rotate[2] * current_pos_1 + rotate[3] * current_pos_0);
        

        // qmul(current_pos, current_pos2, rotate_inverse);
        current_pos_0 = (current_pos2_0 * rotate_inverse_0 - current_pos2_1 * rotate_inverse_1 - current_pos2_2 * rotate_inverse_2 - current_pos2_3 * rotate_inverse_3);
        current_pos_1 = (current_pos2_0 * rotate_inverse_1 + current_pos2_1 * rotate_inverse_0 + current_pos2_2 * rotate_inverse_3 - current_pos2_3 * rotate_inverse_2);
        current_pos_2 = (current_pos2_0 * rotate_inverse_2 - current_pos2_1 * rotate_inverse_3 + current_pos2_2 * rotate_inverse_0 + current_pos2_3 * rotate_inverse_1);
        current_pos_3 = (current_pos2_0 * rotate_inverse_3 + current_pos2_1 * rotate_inverse_2 - current_pos2_2 * rotate_inverse_1 + current_pos2_3 * rotate_inverse_0);
        // std::cout<<current_pos_0<<" "<<current_pos_1<<" "<<current_pos_2<<" "<<current_pos_3<<std::endl;
        s_current_pose[thread_idx*3] = current_pos_1;
        s_current_pose[thread_idx*3+1] = current_pos_2;
        s_current_pose[thread_idx*3+2] = current_pos_3;

        s_current_pose[thread_idx*3] += pos_gloworm[0];
        s_current_pose[thread_idx*3+1] +=  pos_gloworm[1];
        s_current_pose[thread_idx*3+2] +=  pos_gloworm[2];

        current_ligand_pose[n*4] = s_current_pose[thread_idx*3];
        current_ligand_pose[n*4+1] = s_current_pose[thread_idx*3+1];
        current_ligand_pose[n*4+2] = s_current_pose[thread_idx*3+2];
        current_ligand_pose[n*4+3] = 0;

        // reference
        //  blockz 和 blocky
        // if(block_idz==3 && block_idy==3 && block_idx==0 && thread_idx==0){
        //     printf("%f %f %f \n", current_ligand_pose[n*3],current_ligand_pose[n*3+1],current_ligand_pose[n*3+2]);
        // }
        if(thread_idx == 0 && block_idx==0){
            for(int i=0;i<3;i++){
                current_ligand_reference_pose[i] = d_current_ligand_reference_pose[i];
            }
            // rotate
            double current_pos_0 = 0;
            // double w = 0 ;
            double current_pos_1 = current_ligand_reference_pose[0];
            double current_pos_2 = current_ligand_reference_pose[1];
            double current_pos_3 = current_ligand_reference_pose[2];
            // memcpy(current_pos+1, &coord[i*3], sizeof(double)*3);

            // qmul(current_pos2,rotate,current_pos);
            double  current_pos2_0 = (rotate[0] * current_pos_0 - rotate[1] * current_pos_1 - rotate[2] * current_pos_2 - rotate[3] * current_pos_3);
            double current_pos2_1 = (rotate[0] * current_pos_1 + rotate[1] * current_pos_0 + rotate[2] * current_pos_3 - rotate[3] * current_pos_2);
            double current_pos2_2 = (rotate[0] * current_pos_2 - rotate[1] * current_pos_3 + rotate[2] * current_pos_0 + rotate[3] * current_pos_1);
            double current_pos2_3 = (rotate[0] * current_pos_3 + rotate[1] * current_pos_2 - rotate[2] * current_pos_1 + rotate[3] * current_pos_0);
            

            // qmul(current_pos, current_pos2, rotate_inverse);
            current_pos_0 = (current_pos2_0 * rotate_inverse_0 - current_pos2_1 * rotate_inverse_1 - current_pos2_2 * rotate_inverse_2 - current_pos2_3 * rotate_inverse_3);
            current_pos_1 = (current_pos2_0 * rotate_inverse_1 + current_pos2_1 * rotate_inverse_0 + current_pos2_2 * rotate_inverse_3 - current_pos2_3 * rotate_inverse_2);
            current_pos_2 = (current_pos2_0 * rotate_inverse_2 - current_pos2_1 * rotate_inverse_3 + current_pos2_2 * rotate_inverse_0 + current_pos2_3 * rotate_inverse_1);
            current_pos_3 = (current_pos2_0 * rotate_inverse_3 + current_pos2_1 * rotate_inverse_2 - current_pos2_2 * rotate_inverse_1 + current_pos2_3 * rotate_inverse_0);
            // std::cout<<current_pos_0<<" "<<current_pos_1<<" "<<current_pos_2<<" "<<current_pos_3<<std::endl;
            current_ligand_reference_pose[0] = current_pos_1;
            current_ligand_reference_pose[1] = current_pos_2;
            current_ligand_reference_pose[2] = current_pos_3;

            current_ligand_reference_pose[0] += pos_gloworm[0];
            current_ligand_reference_pose[1] += pos_gloworm[1];
            current_ligand_reference_pose[2] += pos_gloworm[2];

            
        }
        // if(block_idz==3 && block_idy==3 && block_idx==0 && thread_idx==0){
        //     printf("%f %f %f \n", current_ligand_reference_pose[0],current_ligand_reference_pose[1],current_ligand_reference_pose[2]);
        // }
            
            
            
    }

}

    

__global__ void calculate_dfire(
    double * d_receptor_pose, double * d_ligand_pose, 
    double * d_scoring_base, double * d_luciferin_base, double * d_rho_base, double * d_gamma_base, 
    double * energy_tmp,
    int * d_moved_base, int * d_step_base,
    double * fastdfire, int * dfire_objects_rec, int * dfire_objects_lig, unsigned int * d_dist_to_bins,
    int swarms, int num_glowworms, int anm_lig, int anm_rec,
    int receptor_atoms, int ligand_atoms, int rec_len, int lig_len
){
    const int block_idx = blockIdx.x;
    const int block_idy = blockIdx.y;
    const int block_idz = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;
    // if(block_idz==3 && block_idy==3 && block_idx==0 && thread_idx==0){
    //         printf("test \n");
    // }
    int pos_len = 7+anm_rec+anm_lig;

    int moved = d_moved_base[block_idz*num_glowworms + block_idy];
    int steped = d_step_base[block_idz*num_glowworms + block_idy];
    int n = block_size * block_idx + thread_idx;
    if((moved || steped == 0) && n < receptor_atoms){

    // __shared__ int s_mem[8192];
        __shared__ double s_mem[1024][3+1+1];
        
        // __shared__ double s_mem[512];
        // double * s_pos = s_mem;
        
        // double * s_current_receptor_pose = (double *)(s_mem);
        int ligand_idx_base = 256;
        int s_lig_idx = ligand_idx_base + thread_idx;

        int * s_dfire_objects_rec = (int * )(&s_mem[513][0]);
        
        
        
        // share 
        double * current_ligand_pose = &d_ligand_pose[block_idz * num_glowworms * lig_len*4 + block_idy * lig_len*4];
        // double * current_ligand_reference_pose = &d_ligand_reference_pose[block_idz * num_glowworms * 3 + block_idy * 3];
        double * current_receptor_pose = &d_receptor_pose[block_idz * num_glowworms * rec_len*4 + block_idy * rec_len*4];

        

        double rho =  d_rho_base[block_idz*num_glowworms+block_idy];
        double luciferins = d_luciferin_base[block_idz*num_glowworms+block_idy];
        double gamma = d_gamma_base[block_idz*num_glowworms+block_idy];
        double energy = d_scoring_base[block_idz*num_glowworms+block_idy];

        
        s_mem[thread_idx][0] = (n<receptor_atoms) ? current_receptor_pose[n*4+0] : 0;
        s_mem[thread_idx][1] = (n<receptor_atoms) ? current_receptor_pose[n*4+1] : 0;
        s_mem[thread_idx][2] = (n<receptor_atoms) ? current_receptor_pose[n*4+2] : 0;
        s_mem[thread_idx][3] = (n<receptor_atoms) ? current_receptor_pose[n*4+3] : 0;
        // s_current_receptor_pose[thread_idx * 3] = current_receptor_pose[n*3+0];
        // s_current_receptor_pose[thread_idx * 3 + 1] = current_receptor_pose[n*3+1];
        // s_current_receptor_pose[thread_idx * 3 + 2] = current_receptor_pose[n*3+2];
        s_dfire_objects_rec[thread_idx] = (n<receptor_atoms) ? dfire_objects_rec[n] : 0;
        // s_ener[thread_idx] = 0;
        __syncthreads();

        int atoma = s_dfire_objects_rec[thread_idx] *168*20;
    
        energy=0.0;
        
        
        for (int j = 0; j < ligand_atoms; j++) {
            double dist0 = s_mem[thread_idx][0] - current_ligand_pose[j*4+0];
            double dist1 = s_mem[thread_idx][1] - current_ligand_pose[j*4+1];
            double dist2 = s_mem[thread_idx][2] - current_ligand_pose[j*4+2];

            double dist = dist0 * dist0 + dist1 * dist1 + dist2 * dist2;
            // if(n >= receptor_atoms) {dist=0.0;}
            
            if (dist <= 225 && dist > 0) {

                int d = (sqrt(dist)*2.0 - 1.0);
                
                int atomb = dfire_objects_lig[j];
                int dfire_bin = d_dist_to_bins[d] - 1;
                unsigned int array_ = atoma + atomb*20 + dfire_bin;
                double value = fastdfire[array_];
                energy += value;
                

            }
        }

        // if(block_idz == 0 && block_idy==0 && block_idx == 0 && thread_idx == 0){
        //     printf("test: %f", energy);
        // }
        

        
        // reduce
        // 当前receptor的enrngy
        
        double * s_ener = (double*)(&s_mem[0][0]);
        s_ener[thread_idx] = (n<receptor_atoms) ? energy : 0;
        __syncthreads();

        for(int offset = block_size >> 1; offset >= 32; offset >>=1){
            if(thread_idx < offset){
                s_ener[thread_idx] += s_ener[thread_idx+offset];
            }
            __syncthreads();
        }
        double y = s_ener[thread_idx];
        for(int offset = 16; offset > 0; offset>>=1){
            y+=__shfl_down_sync(FULL_MASK,y,offset);
        }

        if(thread_idx==0){
            energy_tmp[block_idz*num_glowworms*gridDim.x + block_idy*gridDim.x + block_idx] = y;
        }
    }
}


__global__ void cal_score(
    double * d_scoring_base, double * d_luciferin_base, double * d_rho_base, double * d_gamma_base, 
    double * energy_tmp,
    int * d_moved_base, int * d_step_base,
    int swarms, int num_glowworms, int anm_lig, int anm_rec,
    int receptor_atoms, int ligand_atoms, int len
){
    const int block_idx = blockIdx.x;
    const int block_idy = blockIdx.y;
    // const int block_idz = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;
    __shared__ double  s_ener[512];

    double rho =  d_rho_base[block_idy*num_glowworms+block_idx];
    double luciferins = d_luciferin_base[block_idy*num_glowworms+block_idx];
    double gamma = d_gamma_base[block_idy*num_glowworms+block_idx];
    // double energy = d_scoring_base[block_idy*num_glowworms+block_idx];

    int moved = d_moved_base[block_idy*num_glowworms + block_idx];
    int steped = d_step_base[block_idy*num_glowworms + block_idx];

    // double * current_energy = &energy_tmp[block_idy*num_glowworms*block_size+block_idx*block_size];
    int N = swarms*num_glowworms*len;
    int n = block_idy*num_glowworms*len+block_idx*len+thread_idx;
    __syncthreads();

    // if(block_idy==3 && block_idx==3){

    //     printf("%f \n", energy_tmp[n]);
    // }
    if((moved || steped == 0)){
        s_ener[thread_idx] = (n<N) ? energy_tmp[n]: 0;
        __syncthreads();

        for(int offset = block_size >> 1; offset >= 32; offset >>=1){
            if(thread_idx < offset){
                s_ener[thread_idx] += s_ener[thread_idx+offset];
            }
            __syncthreads();
        }
        double y = s_ener[thread_idx];
        for(int offset = 16; offset > 0; offset>>=1){
            y+=__shfl_down_sync(FULL_MASK,y,offset);
        }

        if(thread_idx==0){
            d_scoring_base[block_idy*num_glowworms+block_idx] = ((y)*0.0157 - 4.7)*-1;
        }
        // if(block_idy == 0 && block_idx==0 && thread_idx==0){
        //     printf("energy %f \n", d_scoring_base[block_idy*num_glowworms+block_idx]);
        // }


    }
    if(thread_idx == 0){
        d_luciferin_base[block_idy*num_glowworms+block_idx] = (1.0 - rho) * d_luciferin_base[block_idy*num_glowworms+block_idx] + gamma * d_scoring_base[block_idy*num_glowworms+block_idx];
    }


}

__global__ void cal_move_neighbors_v2(
    double * d_select_position_base, int * d_neighbors_base, double * d_vision_range_base, int * d_select_base, int * d_nnei_len_base,
    double * d_luciferin_base, double * d_probabilities, double * d_prob_array, double * d_centers_pos, double * d_ligand_reference_pose,
    int swarms, int num_glowworms, int anm_lig, int anm_rec,
    int ligand_atoms,int step
){

    int block_idy = blockIdx.y;
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int ndx = block_idy * num_glowworms + block_idx * blockDim.x + thread_idx;

    double * current_reference = &d_ligand_reference_pose[block_idy*num_glowworms*3+(block_idx * blockDim.x + thread_idx)*3];
    int * neighbors = &d_neighbors_base[block_idy*num_glowworms*num_glowworms + (block_idx * blockDim.x + thread_idx)*num_glowworms];
    double vision_range = d_vision_range_base[block_idy*num_glowworms+block_idx * blockDim.x + thread_idx];
    int nnei_len = 0;
    double * move_postion = &d_select_position_base[block_idy * num_glowworms * (7+anm_rec+anm_lig) + (block_idx * blockDim.x + thread_idx) * (7+anm_rec+anm_lig)];
    double current_luciferin = d_luciferin_base[block_idy*num_glowworms + (block_idx * blockDim.x + thread_idx)];
    double * probabilities = &d_probabilities[block_idy*num_glowworms*num_glowworms+ (block_idx * blockDim.x + thread_idx)*num_glowworms];
    double squared_vision_range = vision_range * vision_range;
    double prob = d_prob_array[block_idy*STEPS*num_glowworms + step * num_glowworms + (block_idx * blockDim.x + thread_idx)];
    double * current_pos = &d_centers_pos[block_idy*num_glowworms*(7+anm_rec+anm_lig)];

    // int select_index = d_select_index[block_idx];
    // int * select = d_select[block_idx*num_glowworms];

    int current_nnei_len = 0;
    // step 7
    if(block_idx==0 && thread_idx==95){
        printf("improtant nl:%.10f\n",d_luciferin_base[block_idx*num_glowworms + thread_idx]); // -53.655593
        printf("%f %f %f \n",current_reference[0],current_reference[1],current_reference[2]); // 1.606625 24.444249 -7.706393 
        printf("%f \n", d_vision_range_base[block_idx*num_glowworms+thread_idx]);  //2.2
        
    }
    // for(int ii = 0; ii< num_glowworms; ii++){
    //     neighbors[ii] = 0;
    // }
    for(int n = 0; n<num_glowworms; n++){
        double neighbor_luciferin = d_luciferin_base[block_idy*num_glowworms + n];
        
        if(n!=thread_idx && current_luciferin < neighbor_luciferin){
            
            double * neighbor_reference = &d_ligand_reference_pose[block_idy*num_glowworms*3 + n*3];
            double distance = 0;
            for(int dis = 0; dis < 3; dis++){
                distance += ((current_reference[dis] - neighbor_reference[dis])*(current_reference[dis] - neighbor_reference[dis]));
            }
            // if(block_idx==0 && thread_idx==56 && n==19){
                   
            //         printf("%f \n",distance);
            // }
            if (distance < squared_vision_range){
                // neighbors 
                neighbors[current_nnei_len++] = n; // param
            }

        }

    }
    
    if(block_idx==0 && thread_idx==95){
        printf("nnei_len %d\n",current_nnei_len);
        for(int ii = 0; ii< current_nnei_len; ii++){
            printf("%d ",neighbors[ii]);
        }
        printf("\n");
    }
    d_nnei_len_base[ndx] = current_nnei_len;


    compute_probability_moving_from_neighbors_gpu(probabilities, neighbors, &d_luciferin_base[block_idy*num_glowworms], current_nnei_len,  current_luciferin);
    //当前进程所选的index
    // 重要，计算sum_probabilities 的前缀加和，根据前缀加和确定
    
    if(current_nnei_len == 0){
        d_select_base[ndx] = thread_idx;
    }else{
        double sum_probabilities = 0;
        int idx = 0;
        while(sum_probabilities < prob){
            sum_probabilities += probabilities[idx];
            idx++;
        }
        d_select_base[ndx] = neighbors[idx-1];
        // if(block_idx==0 && thread_idx==56){
        //     printf("idx %d\n",neighbors[idx-1]);
        //     // printf("move_postion:%f %f %f \n",current_pos[thread_idx*(7+anm_rec+anm_lig)],current_pos[thread_idx*(7+anm_rec+anm_lig)+1],current_pos[thread_idx*(7+anm_rec+anm_lig)+2]);
            
        // }
    }
    int nnei_index = d_select_base[ndx];

    // if(block_idx==0 && thread_idx==95){
    //     printf("nnei_index %d\n",nnei_index);
    //     printf("prob %f \n",prob);
    //     // printf("move_postion:%f %f %f \n",current_pos[thread_idx*(7+anm_rec+anm_lig)],current_pos[thread_idx*(7+anm_rec+anm_lig)+1],current_pos[thread_idx*(7+anm_rec+anm_lig)+2]);
        
    // }

    for(int ii = 0; ii<(7+anm_rec+anm_lig); ii++){
        move_postion[ii] = current_pos[nnei_index*(7+anm_rec+anm_lig) + ii];
    }

    // if(thread_idx==56 && block_idx ==0){
    //     printf("move_postion1111:%f %f %f \n",move_postion[0],move_postion[1],move_postion[2]);
    //     printf("move_translation2:%f %f %f \n",current_pos[nnei_index*(7+anm_rec+anm_lig) + 0],current_pos[nnei_index*(7+anm_rec+anm_lig) + 1],current_pos[nnei_index*(7+anm_rec+anm_lig) + 2]);
    
    // }

}

__global__ void cal_move_neighbors(
    double * d_select_position_base, int * d_neighbors_base, double * d_vision_range_base, int * d_select_base, int * d_nnei_len_base,
    double * d_luciferin_base, double * d_probabilities, double * d_prob_array, double * d_centers_pos, double * d_ligand_reference_pose,
    size_t swarms, size_t num_glowworms, size_t anm_lig, size_t anm_rec,
    size_t ligand_atoms,size_t step
){
    size_t block_idx = blockIdx.x;
    size_t thread_idx = threadIdx.x;

    double * current_reference = &d_ligand_reference_pose[block_idx*num_glowworms*3+thread_idx*3];
    int * neighbors = &d_neighbors_base[block_idx*num_glowworms*num_glowworms + thread_idx*num_glowworms];
    double vision_range = d_vision_range_base[block_idx*num_glowworms+thread_idx];
    int nnei_len = 0;
    double * move_postion = &d_select_position_base[block_idx * num_glowworms * (7+anm_rec+anm_lig) + thread_idx * (7+anm_rec+anm_lig)];
    double current_luciferin = d_luciferin_base[block_idx*num_glowworms + thread_idx];
    double * probabilities = &d_probabilities[block_idx*num_glowworms*num_glowworms+ thread_idx*num_glowworms];
    double squared_vision_range = vision_range * vision_range;
    double prob = d_prob_array[block_idx*STEPS*num_glowworms + step * num_glowworms + thread_idx];
    double * current_pos = &d_centers_pos[block_idx*num_glowworms*(7+anm_rec+anm_lig)];

    // int select_index = d_select_index[block_idx];
    // int * select = d_select[block_idx*num_glowworms];

    int current_nnei_len = 0;
    // step 7
    // if(block_idx==0 && thread_idx==95){
    //     printf("improtant nl:%.10f\n",d_luciferin_base[block_idx*num_glowworms + thread_idx]); // -53.655593
    //     printf("%f %f %f \n",current_reference[0],current_reference[1],current_reference[2]); // 1.606625 24.444249 -7.706393 
    //     printf("%f \n", d_vision_range_base[block_idx*num_glowworms+thread_idx]);  //2.2
        
    // }
    // for(int ii = 0; ii< num_glowworms; ii++){
    //     neighbors[ii] = 0;
    // }
    for(int n = 0; n<num_glowworms; n++){
        double neighbor_luciferin = d_luciferin_base[block_idx*num_glowworms + n];
        
        if(n!=thread_idx && current_luciferin < neighbor_luciferin){
            
            double * neighbor_reference = &d_ligand_reference_pose[block_idx*num_glowworms*3 + n*3];
            double distance = 0;
            for(int dis = 0; dis < 3; dis++){
                distance += ((current_reference[dis] - neighbor_reference[dis])*(current_reference[dis] - neighbor_reference[dis]));
            }
            // if(block_idx==0 && thread_idx==56 && n==19){
                   
            //         printf("%f \n",distance);
            // }
            if (distance < squared_vision_range){
                // neighbors 
                neighbors[current_nnei_len++] = n; // param
            }

        }

    }
    
    // if(block_idx==0 && thread_idx==95){
    //     printf("nnei_len %d\n",current_nnei_len);
    //     for(int ii = 0; ii< current_nnei_len; ii++){
    //         printf("%d ",neighbors[ii]);
    //     }
    //     printf("\n");
    // }
    d_nnei_len_base[block_idx*num_glowworms+thread_idx] = current_nnei_len;


    compute_probability_moving_from_neighbors_gpu(probabilities, neighbors, &d_luciferin_base[block_idx*num_glowworms], current_nnei_len,  current_luciferin);
    //当前进程所选的index
    // 重要，计算sum_probabilities 的前缀加和，根据前缀加和确定
    
    if(current_nnei_len == 0){
        d_select_base[block_idx*num_glowworms+thread_idx] = thread_idx;
    }else{
        double sum_probabilities = 0;
        int idx = 0;
        while(sum_probabilities < prob){
            sum_probabilities += probabilities[idx];
            idx++;
        }
        d_select_base[block_idx*num_glowworms+thread_idx] = neighbors[idx-1];
        // if(block_idx==0 && thread_idx==56){
        //     printf("idx %d\n",neighbors[idx-1]);
        //     // printf("move_postion:%f %f %f \n",current_pos[thread_idx*(7+anm_rec+anm_lig)],current_pos[thread_idx*(7+anm_rec+anm_lig)+1],current_pos[thread_idx*(7+anm_rec+anm_lig)+2]);
            
        // }
    }
    int nnei_index = d_select_base[block_idx*num_glowworms+thread_idx];

    // if(block_idx==0 && thread_idx==95){
    //     printf("nnei_index %d\n",nnei_index);
    //     printf("prob %f \n",prob);
    //     // printf("move_postion:%f %f %f \n",current_pos[thread_idx*(7+anm_rec+anm_lig)],current_pos[thread_idx*(7+anm_rec+anm_lig)+1],current_pos[thread_idx*(7+anm_rec+anm_lig)+2]);
        
    // }

    for(int ii = 0; ii<(7+anm_rec+anm_lig); ii++){
        move_postion[ii] = current_pos[nnei_index*(7+anm_rec+anm_lig) + ii];
    }

    // if(thread_idx==56 && block_idx ==0){
    //     printf("move_postion1111:%f %f %f \n",move_postion[0],move_postion[1],move_postion[2]);
    //     printf("move_translation2:%f %f %f \n",current_pos[nnei_index*(7+anm_rec+anm_lig) + 0],current_pos[nnei_index*(7+anm_rec+anm_lig) + 1],current_pos[nnei_index*(7+anm_rec+anm_lig) + 2]);
    
    // }

}

__device__ __forceinline__
void slerp_gpu(double * self, double * other, double rotation_step){
    // calculate inter polation
    double self_norm2 = 0;
    double other_norm2 = 0;

    for(int i = 0; i<4;i++){
        self_norm2 += (self[i] * self[i]);
        other_norm2 += (other[i] * other[i]);
    }
    self_norm2 = sqrt(self_norm2);
    other_norm2 = sqrt(other_norm2);
    for(int i = 0; i<4;i++){
        self[i] /= self_norm2;
        other[i] /= other_norm2;
    }
    // calc dot
    double q_dot = 0;
    for(int i = 0; i<4;i++){
        q_dot += self[i] * other[i];
    }
    if(q_dot < 0){
        q_dot = q_dot * -1;
        for(int i = 0; i< 4; i++){
            self[i] = -self[i];
        }
    }

    if(q_dot>0.9995){
        for(int i = 0; i < 4; i++){
            self[i] = self[i] + rotation_step * (other[i] - self[i]);
        }
        double result_norm = 0;
        for(int i = 0; i< 4; i++){
            result_norm += (self[i] * self[i]);
        }
        result_norm = sqrt(result_norm);
        for(int i = 0; i<4;i++){
            self[i] /= result_norm;
        }
    }else{
        q_dot = (q_dot < 1) ? q_dot : 1;
        q_dot = (q_dot > -1) ? q_dot : -1;
        double omega = acos(q_dot);
        double so = sin(omega);
        double tmp0 = (sin((1.0-rotation_step)*omega) / so);
        double tmp1 = (sin(rotation_step*omega)/so);
        for(int i = 0; i< 4; i++){
            self[i] = tmp0 * self[i] + tmp1 * other[i];
        }
    }

}

__device__ __forceinline__
void move_extent(double * self, double * other, double * delta_x ,int num_ext,double step_nmodes){
    // double * delta_x = new double [num_ext];
    double n = 0;
    for(int i = 0; i< num_ext; i++){
        delta_x[i] = other[i] - self[i];
        n += delta_x[i] * delta_x[i];
    }   
    
    n = sqrt(n);
    if(n>=1e-8){
        for(int i = 0; i< num_ext; i++){
            delta_x[i] *= step_nmodes/n;
            self[i] += delta_x[i];
        }
    }
    // delete [] delta_x;
}

__global__ void move_position(
    double * d_vision_range_base, double * d_centers_pos, double * d_select_position_base, 
    int * d_id_base, int *d_selece_base, double * d_delta_base, double * d_rec_base, double * d_lig_base, 
    double * d_beta_base, int  * d_max_neighbors_base,
    double * d_max_vision_range_base, int * d_nnei_len_base,
    int * d_moved_base, int swarm, int num_glowworms,int anm_lig, int anm_rec,
    int ligand_atoms,int step
){
    int thread_idx = threadIdx.x;
    int block_idx = blockIdx.x;
    double * current_postion = &d_centers_pos[block_idx*num_glowworms*(7+anm_rec+anm_lig)+thread_idx*(7+anm_rec+anm_lig)];
    double * move_postion = &d_select_position_base[block_idx*num_glowworms*(7+anm_rec+anm_lig)+thread_idx*(7+anm_rec+anm_lig)];
    int id = d_id_base[block_idx*num_glowworms+thread_idx];
    // TODO: 冗余
    int nei_idx = d_selece_base[block_idx*num_glowworms+thread_idx];
    int nei_id = d_id_base[block_idx*num_glowworms+nei_idx];
    double *delta = &d_delta_base[block_idx*num_glowworms*3+thread_idx*3];
    double * rec_ext = &d_rec_base[block_idx*num_glowworms*anm_rec+thread_idx*anm_rec];
    double *lig_ext = &d_lig_base[block_idx*num_glowworms*anm_lig+thread_idx*anm_lig];
    d_moved_base[block_idx*num_glowworms+thread_idx] = id != nei_id;
    // if(thread_idx==62 && block_idx ==0){
    //     printf("%d\n",id);
    //     printf("%d\n",nei_id);
    // }
    if(id != nei_id){
        double * move_tranlation = &move_postion[0];
        double * current_translation = &current_postion[0];
        double * current_rotation = &current_postion[3];
        double * move_rotation = &move_postion[3];

        double * current_rec_extent = &current_postion[7];
        double * move_rec_extent = &move_postion[7];

        double * current_lig_extent = &current_postion[7+anm_rec];
        double * move_lig_extent = &move_postion[7+anm_rec];
        
        double n = 0;
        // if(thread_idx==62 && block_idx ==0){
        //     printf("move_translation:%f %f %f \n",move_tranlation[0],move_tranlation[1],move_tranlation[2]);
        //     printf("move_translation2:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
        
        // }
        // 问题出在这里
        // if(thread_idx==95 && block_idx ==0){
        //     printf("current_translation:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
        //     printf("move:%f %f %f \n",move_tranlation[0],move_tranlation[1],move_tranlation[2]);
        
        // }
        for(int d = 0; d<3; d++){
            delta[d] = move_tranlation[d] - current_translation[d];
            n+=delta[d] * delta[d];
        }
        n = sqrt(n);
        // if(thread_idx==62 && block_idx ==0){
        //     printf("n: %f", n);

        // }
        if(n>=1e-8){
            for(int d = 0; d<3; d++){
                delta[d]*=(0.5 / n);
                current_translation[d] += delta[d];
            }
        }
        
        slerp_gpu(current_rotation,move_rotation, 0.5);

        move_extent(current_rec_extent,move_rec_extent,rec_ext,anm_rec,0.5);
        move_extent(current_lig_extent,move_lig_extent,lig_ext,anm_lig,0.5);
        // if(thread_idx==56 && block_idx ==0){
        //     printf("current_translation:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
        //     printf("current_rotation:%f %f %f %f \n",current_rotation[0],current_rotation[1],current_rotation[2], current_rotation[3]);
        //     printf("rec_ext:%f %f %f %f \n",current_rec_extent[0],current_rec_extent[1],current_rec_extent[2], current_rec_extent[3]);
        //     printf("lig_ext:%f %f %f %f \n",current_lig_extent[0],current_lig_extent[1],current_lig_extent[2], current_lig_extent[3]);
        //     // printf("current_translation:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
        
        // }
    }
    d_vision_range_base[block_idx*num_glowworms+thread_idx] = d_vision_range_base[block_idx*num_glowworms+thread_idx] + d_beta_base[block_idx*num_glowworms+thread_idx] * (d_max_neighbors_base[block_idx*num_glowworms+thread_idx] - d_nnei_len_base[block_idx*num_glowworms+thread_idx]);
    if(d_vision_range_base[block_idx*num_glowworms+thread_idx]<0.0){d_vision_range_base[block_idx*num_glowworms+thread_idx] = 0;}
    if(d_vision_range_base[block_idx*num_glowworms+thread_idx] > d_max_vision_range_base[block_idx*num_glowworms+thread_idx]){d_vision_range_base[block_idx*num_glowworms+thread_idx]=d_max_vision_range_base[block_idx*num_glowworms+thread_idx];}
    // if(thread_idx==56 && block_idx ==0){
    //     printf(" viusion range%f\n",d_vision_range_base[block_idx*num_glowworms+thread_idx]);
    //     // printf("%d\n",nei_id);
    // }

}

__global__ void move_position_v2(
    double * d_vision_range_base, double * d_centers_pos, double * d_select_position_base, 
    int * d_id_base, int *d_selece_base, double * d_delta_base, double * d_rec_base, double * d_lig_base, 
    double * d_beta_base, int  * d_max_neighbors_base,
    double * d_max_vision_range_base, int * d_nnei_len_base,
    int * d_moved_base, size_t swarm, size_t num_glowworms,size_t anm_lig, size_t anm_rec,
    size_t ligand_atoms,size_t step
){
    size_t thread_idx = threadIdx.x;
    size_t block_idx = blockIdx.x;
    size_t block_idy = blockIdx.y;
    size_t ndx = block_idy * num_glowworms + block_idx * blockDim.x + thread_idx;
    double * current_postion = &d_centers_pos[block_idy*num_glowworms*(7+anm_rec+anm_lig)+(block_idx * blockDim.x + thread_idx)*(7+anm_rec+anm_lig)];
    double * move_postion = &d_select_position_base[block_idy*num_glowworms*(7+anm_rec+anm_lig)+(block_idx * blockDim.x + thread_idx)*(7+anm_rec+anm_lig)];
    int id = d_id_base[block_idy*num_glowworms+(block_idx * blockDim.x + thread_idx)];
    // TODO: 冗余
    int nei_idx = d_selece_base[block_idy*num_glowworms+(block_idx * blockDim.x + thread_idx)];
    int nei_id = d_id_base[block_idy*num_glowworms+nei_idx];
    double *delta = &d_delta_base[block_idy*num_glowworms*3+(block_idx * blockDim.x + thread_idx)*3];
    double * rec_ext = &d_rec_base[block_idy*num_glowworms*anm_rec+(block_idx * blockDim.x + thread_idx)*anm_rec];
    double *lig_ext = &d_lig_base[block_idy*num_glowworms*anm_lig+(block_idx * blockDim.x + thread_idx)*anm_lig];
    d_moved_base[block_idy*num_glowworms+(block_idx * blockDim.x + thread_idx)] = id != nei_id;
    // if(thread_idx==62 && block_idx ==0){
    //     printf("%d\n",id);
    //     printf("%d\n",nei_id);
    // }
    if(id != nei_id){
        double * move_tranlation = &move_postion[0];
        double * current_translation = &current_postion[0];
        double * current_rotation = &current_postion[3];
        double * move_rotation = &move_postion[3];

        double * current_rec_extent = &current_postion[7];
        double * move_rec_extent = &move_postion[7];

        double * current_lig_extent = &current_postion[7+anm_rec];
        double * move_lig_extent = &move_postion[7+anm_rec];
        
        double n = 0;

        for(int d = 0; d<3; d++){
            delta[d] = move_tranlation[d] - current_translation[d];
            n+=delta[d] * delta[d];
        }
        n = sqrt(n);
        // if(thread_idx==62 && block_idx ==0){
        //     printf("n: %f", n);

        // }
        if(n>=1e-8){
            for(int d = 0; d<3; d++){
                delta[d]*=(0.5 / n);
                current_translation[d] += delta[d];
            }
        }
        
        slerp_gpu(current_rotation,move_rotation, 0.5);

        move_extent(current_rec_extent,move_rec_extent,rec_ext,anm_rec,0.5);
        move_extent(current_lig_extent,move_lig_extent,lig_ext,anm_lig,0.5);
        // if(thread_idx==56 && block_idx ==0){
        //     printf("current_translation:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
        //     printf("current_rotation:%f %f %f %f \n",current_rotation[0],current_rotation[1],current_rotation[2], current_rotation[3]);
        //     printf("rec_ext:%f %f %f %f \n",current_rec_extent[0],current_rec_extent[1],current_rec_extent[2], current_rec_extent[3]);
        //     printf("lig_ext:%f %f %f %f \n",current_lig_extent[0],current_lig_extent[1],current_lig_extent[2], current_lig_extent[3]);
        //     // printf("current_translation:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
        
        // }
    }
    d_vision_range_base[ndx] = d_vision_range_base[ndx] + d_beta_base[ndx] * (d_max_neighbors_base[ndx] - d_nnei_len_base[ndx]);
    if(d_vision_range_base[ndx]<0.0){d_vision_range_base[ndx] = 0;}
    if(d_vision_range_base[ndx] > d_max_vision_range_base[ndx]){d_vision_range_base[ndx]=d_max_vision_range_base[ndx];}
    // if(thread_idx==56 && block_idx ==0){
    //     printf(" viusion range%f\n",d_vision_range_base[block_idx*num_glowworms+thread_idx]);
    //     // printf("%d\n",nei_id);
    // }

}


template<typename Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__(BLOCK_THREADS)
__global__ void BlockSortKernel(Key * d_in, Key * d_out){
    // using namespace cute;
    enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef cub::BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    // Specialize BlockRadixSort type for our thread block
    typedef cub::BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    // Shared memory
    __shared__ union TempStorage
    {
        typename BlockLoadT::TempStorage        load;
        typename BlockRadixSortT::TempStorage   sort;
    } temp_storage;
    // Per-thread tile items
    Key items[ITEMS_PER_THREAD];
    // Our current block's offset
    int block_offset = blockIdx.x * TILE_SIZE;
    // Load items into a blocked arrangement
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
    // Barrier for smem reuse
    __syncthreads();
    // Sort keys
    BlockRadixSortT(temp_storage.sort).SortDescendingBlockedToStriped(items);
    // Store output in striped fashion
    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset, items);
}

__global__ void prepare_pair(int64_t * d_sort_pair, int* id, double * d_scoring_base, int num_glowworms){
    const int block_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    __shared__ int smem[9216];
    double * s_score = (double *)smem;
    int * s_id = (int *)(smem + 512 * 2);
    int64_t * s_sort_pair = (int64_t*)(smem + 512 * 2 + 512 );
    s_id[thread_idx] = id[block_idx * num_glowworms + thread_idx];
    s_score[thread_idx] = d_scoring_base[block_idx * num_glowworms + thread_idx];
    // mem.s_id[thread_idx] = id[block_idx * num_glowworms + thread_idx];
    // mem.s_score[thread_idx] = d_scoring_base[block_idx * num_glowworms + thread_idx];
    __syncthreads();
    int cur_id = s_id[thread_idx];
    double cur_score = s_score[thread_idx];

    s_sort_pair[thread_idx] = (int64_t)(cur_score * ((int64_t)1<<50)) / (1<<12) * (1<<12) + cur_id;
    d_sort_pair[block_idx * num_glowworms + thread_idx] = s_sort_pair[thread_idx];
}

bool compare2(int64_t g1, int64_t g2){
    return g1>g2;
}

__global__ void perform_rmsd_cal(
    double * d_rmsd_tmp, int * d_backbone_tmp,
    double * d_receptor_pose, double * d_ligand_pose,
    int * cluster, int * rec_name, int *lig_name,
    int num_glowworms, int swarms, int current_g, int num_atoms_rec, int num_atoms_lig
){
    const int block_idx = blockIdx.x;
    const int block_idy = blockIdx.y;
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;
    __shared__ int smem[2048];
    int current_cluster = cluster[block_idy*num_glowworms];
    double * cluster_coord_rec = &d_receptor_pose[swarms * num_glowworms * num_atoms_rec * 3 + current_cluster * num_atoms_rec * 3];
    double * cluster_coord_lig = &d_ligand_pose[swarms * num_glowworms * num_atoms_lig * 3 + current_cluster * num_atoms_lig * 3];

    double * glowworms_coord_rec = &d_receptor_pose[swarms * num_glowworms * num_atoms_rec * 3 + current_g * num_atoms_rec * 3];
    double * glowworms_coord_lig = &d_ligand_pose[swarms * num_glowworms * num_atoms_lig * 3 + current_g * num_atoms_lig * 3];

    int n = block_idx * block_size + thread_idx;
    int N = num_atoms_rec + num_atoms_lig;
    double rmsd = 0;
    int backbone_len = 0;
    if(n<num_atoms_rec){
        rmsd += (cluster_coord_rec[n*3] - glowworms_coord_rec[n*3]) * (cluster_coord_rec[n*3] - glowworms_coord_rec[n*3]);
        rmsd += (cluster_coord_rec[n*3+1] - glowworms_coord_rec[n*3+1]) * (cluster_coord_rec[n*3+1] - glowworms_coord_rec[n*3+1]);
        rmsd += (cluster_coord_rec[n*3+2] - glowworms_coord_rec[n*3+2]) * (cluster_coord_rec[n*3+2] - glowworms_coord_rec[n*3+2]);
        backbone_len ++;
        rmsd *= rec_name[n];
        backbone_len *= rec_name[n];
    }
    if(n<N && n>=num_atoms_rec){
        int idx = n - num_atoms_rec;
        rmsd += (cluster_coord_lig[idx*3] - glowworms_coord_lig[idx*3]) * (cluster_coord_lig[idx*3] - glowworms_coord_lig[idx*3]);
        rmsd += (cluster_coord_lig[idx*3+1] - glowworms_coord_lig[idx*3+1]) * (cluster_coord_lig[idx*3+1] - glowworms_coord_lig[idx*3+1]);
        rmsd += (cluster_coord_lig[idx*3+2] - glowworms_coord_lig[idx*3+2]) * (cluster_coord_lig[idx*3+2] - glowworms_coord_lig[idx*3+2]);
        backbone_len++;
        rmsd *= lig_name[idx];
        backbone_len *= lig_name[idx];
    }

    // block_reduce
    int * s_backbone_len = (int *)smem;
    double * s_rmsd = (double *)(smem + 1024);

    s_backbone_len[thread_idx] = (n<N)? backbone_len : 0;
    s_rmsd[thread_idx] = (n<N)? rmsd : 0;
    __syncthreads();

    for(int offset = block_size >> 1; offset >= 32; offset >>=1){
        if(thread_idx < offset){
            s_rmsd[thread_idx] += s_rmsd[thread_idx+offset];
            s_backbone_len[thread_idx] += s_backbone_len[thread_idx+offset];
        }
        __syncthreads();
    }

    double y = s_rmsd[thread_idx];
    double z = s_backbone_len[thread_idx];
    for(int offset = 16; offset > 0; offset>>=1){
        y+=__shfl_down_sync(FULL_MASK,y,offset);
        z+=__shfl_down_sync(FULL_MASK,z,offset);
    }
    if(thread_idx == 0){
        d_rmsd_tmp[block_idy * gridDim.x + block_idx] = y;
        d_backbone_tmp[block_idy * gridDim.x + block_idx] = z;
    }

}

__global__ void cal_final_rmsd(
    double * d_rmsd, double * d_rmsd_tmp, int * d_backbone_tmp, int num_glowworms, int len
){
    // const int block_idy = blockIdx.y;
    const int block_idx = blockIdx.x;
    const int grid_x = gridDim.x;
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;
    __shared__ int smem[2048];
    int * s_backup_tmp = (int *)smem;
    double * s_rmsd_tmp = (double *)(smem + 1024);
    int n = block_idx * len + thread_idx;
    int N = num_glowworms * len;
    s_backup_tmp[thread_idx] = (n<N)? d_backbone_tmp[n] : 0;
    s_rmsd_tmp[thread_idx] = (n<N)? d_rmsd_tmp[n] : 0;
    __syncthreads();

    for(int offset = block_size >> 1; offset >= 32; offset >>=1){
        if(thread_idx < offset){
            s_rmsd_tmp[thread_idx] += s_rmsd_tmp[thread_idx+offset];
            s_backup_tmp[thread_idx] += s_backup_tmp[thread_idx+offset];
        }
        __syncthreads();
    }

    double y = s_rmsd_tmp[thread_idx];
    int z = s_backup_tmp[thread_idx];
    for(int offset = 16; offset > 0; offset>>=1){
        y+=__shfl_down_sync(FULL_MASK,y,offset);
        z+=__shfl_down_sync(FULL_MASK,z,offset);
    }

    if(thread_idx==0){
        d_rmsd[block_idx] = sqrt(y/z);
    }

}


__global__ void sum_ener(
    double * energy_tmp, double * dist_tmp,
    double * d_receptor_pose, double * d_ligand_pose, 
    double * d_scoring_base, double * d_luciferin_base, double * d_rho_base, double * d_gamma_base, 
    int * d_moved_base, int * d_step_base,
    double * fastdfire, int * dfire_objects_rec, int * dfire_objects_lig, unsigned int * d_dist_to_bins,
    int swarms, int num_glowworms, int anm_lig, int anm_rec,
    int receptor_atoms, int ligand_atoms, int rec_len, int lig_len
){
    const int block_idx = blockIdx.x;
    const int block_idy = blockIdx.y;
    const int block_idz = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;

    int pos_len = 7+anm_rec+anm_lig;

    int moved = d_moved_base[block_idz*num_glowworms + block_idy];
    int steped = d_step_base[block_idz*num_glowworms + block_idy];
    int n = block_size * block_idx + thread_idx;
    if((moved || steped == 0) && n < receptor_atoms){
        // __shared__ int s_mem[8192];
        __shared__ double s_mem[1024][3+1+1];
        
        // __shared__ double s_mem[512];
        // double * s_pos = s_mem;
        
        // double * s_current_receptor_pose = (double *)(s_mem);
        int ligand_idx_base = 256;
        int s_lig_idx = ligand_idx_base + thread_idx;

        int * s_dfire_objects_rec = (int * )(&s_mem[513][0]);
        
        
        
        // share 
        double * current_ligand_pose = &d_ligand_pose[block_idz * num_glowworms * lig_len*4 + block_idy * lig_len*4];
        // double * current_ligand_reference_pose = &d_ligand_reference_pose[block_idz * num_glowworms * 3 + block_idy * 3];
        double * current_receptor_pose = &d_receptor_pose[block_idz * num_glowworms * rec_len*4 + block_idy * rec_len*4];
        // double * current_energy_tmp = &energy_tmp[block_idz * num_glowworms * rec_len + block_idy * rec_len];
        double * current_dist_tmp = &dist_tmp[block_idz * num_glowworms * rec_len * lig_len + block_idy * rec_len * lig_len];
        double * current_ener = &energy_tmp[block_idz * num_glowworms * rec_len + block_idy * rec_len];
        

        double rho =  d_rho_base[block_idz*num_glowworms+block_idy];
        double luciferins = d_luciferin_base[block_idz*num_glowworms+block_idy];
        double gamma = d_gamma_base[block_idz*num_glowworms+block_idy];
        double energy = d_scoring_base[block_idz*num_glowworms+block_idy];

        int atoma = dfire_objects_rec[n] *168*20;
        energy = 0;
        for (int j = 0; j < ligand_atoms; j++) {
            double dist = current_dist_tmp[n * lig_len + j];

            
            if (dist <= 225 && dist > 0) {

                int d = (sqrt(dist)*2.0 - 1.0);
                
                int atomb = dfire_objects_lig[j];
                int dfire_bin = d_dist_to_bins[d] - 1;
                unsigned int array_ = atoma + atomb*20 + dfire_bin;
                double value = fastdfire[array_];
                energy += value;
                

            }

        }
        // energy_tmp = energy;
        current_ener[n] = energy;

        // double * s_ener = (double*)(&s_mem[0][0]);
        // s_ener[thread_idx] = (n<receptor_atoms) ? energy : 0;
        // __syncthreads();

        // for(int offset = block_size >> 1; offset >= 32; offset >>=1){
        //     if(thread_idx < offset){
        //         s_ener[thread_idx] += s_ener[thread_idx+offset];
        //     }
        //     __syncthreads();
        // }
        // double y = s_ener[thread_idx];
        // for(int offset = 16; offset > 0; offset>>=1){
        //     y+=__shfl_down_sync(FULL_MASK,y,offset);
        // }

        // if(thread_idx==0){
        //     energy_tmp[block_idz*num_glowworms*gridDim.x + block_idy*gridDim.x + block_idx] = y;
        // }



    }
}


__global__ void calculate_dfire2(
    double * energy_tmp, double * sum_ener,
    double * d_receptor_pose, double * d_ligand_pose, 
    double * d_scoring_base, double * d_luciferin_base, double * d_rho_base, double * d_gamma_base, 
    int * d_moved_base, int * d_step_base,
    double * fastdfire, int * dfire_objects_rec, int * dfire_objects_lig, unsigned int * d_dist_to_bins,
    int swarms, int num_glowworms, int anm_lig, int anm_rec,
    int receptor_atoms, int ligand_atoms, int rec_len, int lig_len
){
    const int block_idx = blockIdx.x;
    const int block_idy = blockIdx.y;
    const int block_idz = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;
    // if(block_idz==3 && block_idy==3 && block_idx==0 && thread_idx==0){
    //         printf("test \n");
    // }
    int pos_len = 7+anm_rec+anm_lig;

    int moved = d_moved_base[block_idz*num_glowworms + block_idy];
    int steped = d_step_base[block_idz*num_glowworms + block_idy];
    int n = block_size * block_idx + thread_idx;
    if((moved || steped == 0) && n < receptor_atoms){

    // __shared__ int s_mem[8192];
        __shared__ double s_mem[1024][3+1+1];
        
        // __shared__ double s_mem[512];
        // double * s_pos = s_mem;
        
        // double * s_current_receptor_pose = (double *)(s_mem);
        int ligand_idx_base = 256;
        int s_lig_idx = ligand_idx_base + thread_idx;

        // int * s_dfire_objects_rec = (int * )(&s_mem[513][0]);
        
        
        
        // share 
        double * current_ligand_pose = &d_ligand_pose[block_idz * num_glowworms * lig_len*4 + block_idy * lig_len*4];
        // double * current_ligand_reference_pose = &d_ligand_reference_pose[block_idz * num_glowworms * 3 + block_idy * 3];
        double * current_receptor_pose = &d_receptor_pose[block_idz * num_glowworms * rec_len*4 + block_idy * rec_len*4];
        double * current_energy_tmp = &energy_tmp[block_idz * num_glowworms * rec_len * lig_len + block_idy * rec_len * lig_len];
        double * current_ener = &sum_ener[block_idz * num_glowworms * rec_len + block_idy * rec_len];

        double rho =  d_rho_base[block_idz*num_glowworms+block_idy];
        double luciferins = d_luciferin_base[block_idz*num_glowworms+block_idy];
        double gamma = d_gamma_base[block_idz*num_glowworms+block_idy];
        double energy = d_scoring_base[block_idz*num_glowworms+block_idy];

        int atoma = dfire_objects_rec[n] *168*20;
        energy = 0;
        
        
        for (int j = 0; j < ligand_atoms; j++) {
            double dist0 = current_receptor_pose[n*4+0] - current_ligand_pose[j*4+0];
            double dist1 = current_receptor_pose[n*4+1] - current_ligand_pose[j*4+1];
            double dist2 = current_receptor_pose[n*4+2] - current_ligand_pose[j*4+2];

            double dist = dist0 * dist0 + dist1 * dist1 + dist2 * dist2;
            // if(n >= receptor_atoms) {dist=0.0;}
            double tmp = 0;
            // tmp = (sqrt(dist)*2.0 - 1.0);
            if (dist <= 225 && dist > 0) {

                int d = (sqrt(dist)*2.0 - 1.0);
                
                // tmp = d;
                int atomb = dfire_objects_lig[j];
                int dfire_bin = d_dist_to_bins[d] - 1;
                unsigned int array_ = atoma + atomb*20 + dfire_bin;
                double value = fastdfire[array_];
                tmp = value;
                // tmp = value;
                energy += value;
                

            }
            current_energy_tmp[n * lig_len + j] = tmp;

        }
        // energy_tmp = energy;
        current_ener[n] = energy;
    }
}

namespace config{
    using namespace cute;
    template <int kTileM_ = 32, int kTileN_ = 32, int kTileK_ = 4>
    struct GemmConfig{
        // using T = T_;
        static constexpr int kTileM = kTileM_;
        static constexpr int kTileN = kTileN_;
        static constexpr int kTileK = kTileK_;



        using mma_op = SM80_8x8x4_F64F64F64F64_TN;
        using mma_traits = MMA_Traits<mma_op>;
        // 的到mma_atom 32 个线程
        using mma_atom = MMA_Atom<mma_traits>;
        using mma_atom_shape = mma_traits::Shape_MNK;
        using MMA = decltype(make_tiled_mma(mma_atom{}, 
                        make_layout(Shape<_2, _2, _1>{}), 
                        make_layout(Shape<_2, _2, _1>{})));
        using g2s_copy_op = SM80_CP_ASYNC_CACHEALWAYS<double>;
        using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
        using g2s_copy_atom = Copy_Atom<g2s_copy_traits, double>;

        using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}),
                                            make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<1>{}))));
        
        using G2SCopyB = G2SCopyA;


        using S2GCopyAtomC = Copy_Atom<UniversalCopy<double>, double>;
        // using s2g_copy_traits = Copy_Traits<s2g_copy_op>;
        // using s2g_copy_atom = Copy_Atom<s2g_copy_traits, double>;

        using S2GCopyC = 
        decltype(make_tiled_copy(S2GCopyAtomC{},
                                make_layout(make_shape(Int<32>{}, Int<4>{}),
                                            make_stride(Int<4>{}, Int<1>{})),
                                make_layout(make_shape(Int<1>{}, Int<8>{}))));

        using s2r_copy_op = DefaultCopy;
        using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
        using s2r_copy_atom = Copy_Atom<s2r_copy_traits, double>;

        using S2RCopyAtomA = s2r_copy_atom;
        using S2RCopyAtomB = s2r_copy_atom;
        static constexpr int block = size(MMA{});
    };
}


#define FETCH_DOUBLE4(pointer) (reinterpret_cast<double4*>(&(pointer))[0])
template <typename Config, int BLOCK_THREADS, int ITEM_PER_THREAD>
__global__ void 
calculate_dfire_tensorcore_cute2(
    double * d_energy_tmp, 
    double * d_receptor_pose, double * d_ligand_pose, 
    double * d_scoring_base, double * d_luciferin_base, double * d_rho_base, double * d_gamma_base, 
    int * d_moved_base, int * d_step_base,
    double * d_fastdfire,  int * rec_object, int * lig_object, unsigned int * d_dist_to_bins,
    size_t swarms, size_t num_glowworms, 
    size_t receptor_atoms, size_t ligand_atoms, size_t rec_tile, size_t lig_tile, size_t rec_real_len, size_t lig_real_len
){
    using namespace cute;
    using namespace cub;
    using WarpReduce = cub::WarpReduce<double>;
    typedef BlockRadixSort<uint64_t, BLOCK_THREADS, ITEM_PER_THREAD>  BlockRadixSort;
    
    // using X = Underscore;
    
    using TiledMMA = typename Config::MMA;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using S2GCopyC = typename Config::S2GCopyC;
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;

    constexpr size_t kTileM = Config::kTileM;
    constexpr size_t kTileN = Config::kTileN;
    constexpr size_t kTileK = Config::kTileK;
    
    // block 3 
    const size_t block_idx = blockIdx.x;
    const size_t block_idy = blockIdx.y;
    const size_t block_idz = blockIdx.z;
    const size_t thread_idx = threadIdx.x;
    const size_t block_size = blockDim.x;
    const size_t warp_id = thread_idx / 32;
    const size_t lane_id = thread_idx % 32;

    __shared__ double s_mem[6000];
    // __shared__ typename WarpReduce::TempStorage temp_storage[4];
    // __shared__ typename BlockRadixSort::TempStorage temp_sort;
    
    double * Ashm = s_mem;
    double * Bshm = s_mem + 128;
    double * Cshm = s_mem + 256;
    // int * Fshm = (int *)(s_mem + 1281);
    double * Rshm = Ashm + 128;

    double * current_reduce = &d_energy_tmp[block_idz * num_glowworms * rec_tile * lig_tile  + block_idy * rec_tile * lig_tile  + block_idx ];
    

    double * current_ligand_pose = &d_ligand_pose[block_idz * num_glowworms * ligand_atoms * 4 + block_idy * ligand_atoms * 4];
    double * current_receptor_pose = &d_receptor_pose[block_idz * num_glowworms * receptor_atoms * 4 + block_idy * receptor_atoms * 4];
    // double * current_D_tmp = &D_tmp[block_idz * receptor_atoms * ligand_atoms];
    int moved = d_moved_base[block_idz * num_glowworms + block_idy];
    int steped = d_step_base[block_idz * num_glowworms + block_idy];

    double rho =  d_rho_base[block_idz * num_glowworms + block_idy];
    double luciferins = d_luciferin_base[block_idz * num_glowworms + block_idy];
    double gamma = d_gamma_base[block_idz * num_glowworms + block_idy];

    // double * energy_ptr = &d_energy_tmp[block_idz * num_glowworms * receptor_atoms * ligand_atoms + block_idy * receptor_atoms * ligand_atoms];
    // printf("test \n");
    // double energy = d_scoring_base[block_idz * block_sizey * block_size + block_idy * block_size];
    if((moved || steped == 0)){

    // global memory tensor
    Tensor A = make_tensor(make_gmem_ptr(current_receptor_pose), make_shape(receptor_atoms, 4), make_stride(4, Int<1>{}));
    // B layout 为行优先，tensor行加1stride加K
    Tensor B = make_tensor(make_gmem_ptr(current_ligand_pose), make_shape(ligand_atoms, 4), make_stride(4, Int<1>{}));

    Tensor Rec_index = make_tensor(make_gmem_ptr(rec_object), make_shape(receptor_atoms,1), make_stride(Int<1>{}, Int<1>{}));
    Tensor Lig_index = make_tensor(make_gmem_ptr(lig_object), make_shape(ligand_atoms,1),make_stride(Int<1>{}, Int<1>{}));
   
    // 首先对全局内存中的A划分tile ，tile的维度是M，K， 接下来根据block确定哪一个块
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(block_idx / lig_tile, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(block_idx % lig_tile, _));
    // Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(block_idx / lig_tile, block_idx % lig_tile));
    // Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(block_idy, block_idx));
    // Tensor block_dfire = local_tile(dfire_energy, make_tile(Int<kTileM>{}, Int<kTileN*20>{}), make_coord(block_idx / lig_tile, (block_idx % lig_tile) * 20));
    // 8次copy
    Tensor gRec = local_tile(Rec_index, make_tile(Int<kTileM>{}, Int<1>{}), make_coord(block_idx / lig_tile, _));
    Tensor gLig = local_tile(Lig_index, make_tile(Int<kTileN>{}, Int<1>{}), make_coord(block_idx % lig_tile, _));


    // shared memory tensor
    auto sA = make_tensor(make_smem_ptr(Ashm),
                            make_shape(Int<kTileM>{}, Int<kTileK>{},Int<1>{}), make_stride(Int<kTileK>{}, Int<1>{}, Int<1>{}));  
    auto sB = make_tensor(make_smem_ptr(Bshm),
                            make_shape(Int<kTileN>{}, Int<kTileK>{},Int<1>{}), make_stride(Int<kTileK>{}, Int<1>{}, Int<1>{}));  
    auto sC = make_tensor(make_smem_ptr(Cshm),
                            make_shape(Int<kTileM>{}, Int<kTileN>{},Int<1>{}), make_stride(Int<kTileN>{}, Int<1>{}, Int<1>{}));
    // auto sF = make_tensor(make_smem_ptr(Fshm),
    //                         make_shape(Int<4>{}, Int<kTileN*20>{}, Int<1>{}), make_stride(Int<kTileN*20>{}, Int<1>{}, Int<1>{}));
    
    // 根据mma划分tile
    TiledMMA tiled_mma;
    // 对shmem划分到线程
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCsC = thr_mma.partition_C(sC);  // (MMA, MMA_M, MMA_N)
    auto tAsA = thr_mma.partition_A(sA);
    auto tBsB = thr_mma.partition_B(sB);
    // 对寄存器划分到线程
    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(sC(_, _, 0));     // (MMA, MMA_M, MMA_N)
    // auto tCrD = thr_mma.partition_fragment_C(gD(_, _));     // (MMA, MMA_M, MMA_N)
    // auto tAsA =  thr_mma.partition_fragment_A(sA(_, _, 0));
    // auto tBsB =  thr_mma.partition_fragment_A(sB(_, _, 0));
    
    clear(tCrC);

    // 每个线程到共享内存拷贝划分
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(thread_idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(thread_idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy =
        g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

    S2GCopyC s2g_tiled_copy_c;

    // 异步拷贝
    // gmem-> smem
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, 0), tAsA_copy(_, _, _, 0));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, 0), tBsB_copy(_, _, _, 0));
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // smem -> reg
    cute::copy(tAsA(_, _, _, 0), tArA);
    cute::copy(tBsB(_, _, _, 0), tBrB);
    // cute::copy(tCrD, tCgD);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);

    // reg -> smem

    cute::copy(tCrC, tCsC(_,_,_,0));

    __syncthreads();

    // thread par
    Tensor gRow = local_tile(sA, make_shape(Int<1>{}, Int<kTileK>{}, Int<1>{}), make_coord(thread_idx / 4, _, 1));
    // 4 * 32
    Tensor gCol = local_tile(sB, make_shape(Int<kTileK*2>{}, Int<kTileK>{}, Int<1>{}), make_coord(thread_idx % 4, _, 1));
    // 对C进行分块
    Tensor g_tile = local_tile(sC, make_shape(Int<1>{}, Int<kTileK*2>{}, Int<1>{}), make_coord(thread_idx / 4, thread_idx % 4, 1));

    // Tensor gm_tile = local_tile(gC, make_shape(Int<1>{}, Int<kTileK*2>{}), make_coord(thread_idx / 4, thread_idx % 4));

    // Tensor sthr_F = local_tile(sF, make_shape(Int<1>{}, Int<20>{}, Int<1>{}), make_coord(thread_idx/32, (thread_idx%32) * 20, 1));

    // Tensor thr_C = local_tile(gC, make_shape(Int<1>{}, Int<kTileK*2>{}), make_coord(thread_idx / 4, thread_idx % 4));
    // global memory
    Tensor thr_Rec_index = local_tile(gRec, make_shape(Int<1>{},Int<1>{}), make_coord(thread_idx / 4,_));
    Tensor thr_Lig_index = local_tile(gLig, make_shape(Int<kTileK*2>{},Int<1>{}),make_coord(thread_idx % 4,_));

    // Tensor 
    

    auto R_fragment = make_tensor<double>(Shape<_1,_4>{});
    // auto gCr = make_tensor<double>(Shape<_1,_4>{});
    auto C_fragment = make_tensor<double>(Shape<_8,_4>{});

    auto til_fragment = make_tensor<double>(Shape<_1,_8>{});

    auto r_ener = make_tensor<double>(Shape<_1,_1>{});

    auto r_index = make_tensor<int>(Shape<_1,_1>{});
    auto l_index = make_tensor<int>(Shape<_8,_1>{});
    // gmem -> reg
    cute::copy(thr_Rec_index, r_index);
    cute::copy(thr_Lig_index, l_index);
    

    // smem -> reg
    cute::copy(gRow,R_fragment);
    cute::copy(gCol,C_fragment);
    cute::copy(g_tile,til_fragment);

    int rec_num = (block_idx / lig_tile) * kTileM + thread_idx / 4;
    int lig_num = (block_idx % lig_tile) * kTileN + (thread_idx % 4) * 8;
    
#pragma unroll 
    for(int i = 0; i<8;i++){
        til_fragment(i) = til_fragment(i) * -2;
    }
    
    double sum = 0.0;
#pragma unroll
    for(int i = 0; i < 4; i++){
        sum = sum + R_fragment(i) * R_fragment(i);
    }

#pragma unroll
    for(int i = 0; i < 8; i++){
        if (i+lig_num >= lig_real_len){
            sum = 0.0;
        }
        til_fragment(i) = til_fragment(i) + sum;
    }


// take 4 times


// 时钟周期，需要优化 4 8？
    auto col_sum_fragment = make_tensor<double>(Shape<_1,_8>{});

    col_sum_fragment(0) = (C_fragment(0,0) * C_fragment(0,0) + (C_fragment(0,1) * C_fragment(0,1) + (C_fragment(0,2) * C_fragment(0,2) + (C_fragment(0,3) * C_fragment(0,3)))));
    col_sum_fragment(1) = (C_fragment(1,0) * C_fragment(1,0) + (C_fragment(1,1) * C_fragment(1,1) + (C_fragment(1,2) * C_fragment(1,2) + (C_fragment(1,3) * C_fragment(1,3)))));
    col_sum_fragment(2) = (C_fragment(2,0) * C_fragment(2,0) + (C_fragment(2,1) * C_fragment(2,1) + (C_fragment(2,2) * C_fragment(2,2) + (C_fragment(2,3) * C_fragment(2,3)))));
    col_sum_fragment(3) = (C_fragment(3,0) * C_fragment(3,0) + (C_fragment(3,1) * C_fragment(3,1) + (C_fragment(3,2) * C_fragment(3,2) + (C_fragment(3,3) * C_fragment(3,3)))));
    col_sum_fragment(4) = (C_fragment(4,0) * C_fragment(4,0) + (C_fragment(4,1) * C_fragment(4,1) + (C_fragment(4,2) * C_fragment(4,2) + (C_fragment(4,3) * C_fragment(4,3)))));
    col_sum_fragment(5) = (C_fragment(5,0) * C_fragment(5,0) + (C_fragment(5,1) * C_fragment(5,1) + (C_fragment(5,2) * C_fragment(5,2) + (C_fragment(5,3) * C_fragment(5,3)))));
    col_sum_fragment(6) = (C_fragment(6,0) * C_fragment(6,0) + (C_fragment(6,1) * C_fragment(6,1) + (C_fragment(6,2) * C_fragment(6,2) + (C_fragment(6,3) * C_fragment(6,3)))));
    col_sum_fragment(7) = (C_fragment(7,0) * C_fragment(7,0) + (C_fragment(7,1) * C_fragment(7,1) + (C_fragment(7,2) * C_fragment(7,2) + (C_fragment(7,3) * C_fragment(7,3)))));

    if(rec_num >= rec_real_len){
#pragma unroll
        for(int i = 0; i<8; i++){
            col_sum_fragment(i) = 0.0;
        }
    }
    axpby(1,col_sum_fragment,1, til_fragment);
    // __syncthreads();

    int reg_dist_to_bins[50] = {
         1,  1,  1,  2,  3,  4,  5,  6,  7,  8,
         9, 10, 11, 12, 13, 14, 14, 15, 15, 16,
        16, 17, 17, 18, 18, 19, 19, 20, 20, 21,
        21, 22, 22, 23, 23, 24, 24, 25, 25, 26,
        26, 27, 27, 28, 28, 29, 29, 30, 30, 31};
    int atoma = r_index(0) * 168 * 20;

    uint64_t dist_array[8];
    
    double thr_energy = 0;
#pragma unroll
    for(int i = 0; i<8;i++){
        // double dist = til_fragment(i);
        double tmp = 0;
        if(til_fragment(i) <= 225 && til_fragment(i) > 0){
            int d = (sqrt(til_fragment(i))*2.0 - 1.0);
            int dfire_bin = reg_dist_to_bins[d] - 1;
            int atomb = l_index(i) * 20;
            unsigned int _array = atoma + atomb + dfire_bin;
            tmp = d_fastdfire[_array];

        }
        thr_energy += tmp;
    }
    Rshm[thread_idx] = thr_energy;
    __syncthreads();
    for(int offset = block_size >> 1; offset >= 32; offset >>=1){
        if(thread_idx < offset){
            Rshm[thread_idx] += Rshm[thread_idx+offset];
        }
        __syncthreads();
    }
    double y = Rshm[thread_idx];
    
    for(int offset = 16; offset > 0; offset>>=1){
        y+=__shfl_down_sync(FULL_MASK,y,offset);
    }
    if(thread_idx ==0){
        current_reduce[thread_idx] = y;
    }

    
    }

}


template <typename Config, int BLOCK_THREADS, int ITEM_PER_THREAD>
__global__ void 
calculate_dfire_tensorcore_cute3(
    double * d_energy_tmp, 
    double * d_receptor_pose, double * d_ligand_pose, 
    double * d_scoring_base, double * d_luciferin_base, double * d_rho_base, double * d_gamma_base, 
    int * d_moved_base, int * d_step_base,
    double * d_fastdfire,  int * rec_object, int * lig_object, unsigned int * d_dist_to_bins,
    size_t swarms, size_t num_glowworms, 
    size_t receptor_atoms, size_t ligand_atoms, size_t rec_tile, size_t lig_tile, size_t rec_real_len, size_t lig_real_len
){
    using namespace cute;
    using namespace cub;
    using WarpReduce = cub::WarpReduce<double>;
    // typedef BlockRadixSort<uint64_t, BLOCK_THREADS, ITEM_PER_THREAD>  BlockRadixSort;
    
    // using X = Underscore;
    
    using TiledMMA = typename Config::MMA;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using S2GCopyC = typename Config::S2GCopyC;
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;

    constexpr size_t kTileM = Config::kTileM;
    constexpr size_t kTileN = Config::kTileN;
    constexpr size_t kTileK = Config::kTileK;
    
    // block 3 
    const size_t block_idx = blockIdx.x;
    const size_t block_idy = blockIdx.y;
    const size_t block_idz = blockIdx.z;
    const size_t thread_idx = threadIdx.x;
    const size_t block_size = blockDim.x;
    const size_t warp_id = thread_idx / 32;
    const size_t lane_id = thread_idx % 32;
    const size_t sub_lan = lane_id % 8;
    // 512 + 256
    __shared__ double s_mem[768];
    // __shared__ typename BlockRadixSort::TempStorage temp_sort;
    
    double * Ashm = s_mem;
    double * Bshm = s_mem + 256;
    double * Rshm = s_mem;
    // int srec_idx = (int *)(s_mem+512);
    // int slig_idx = (int *)(s_mem+512 + 32);
    // double * Rshm = s_mem;
    // int * Fshm = (int *)(s_mem + 4680);
    // double * Rshm = (double *)(Fshm + 128);

    double * current_reduce = &d_energy_tmp[block_idz * num_glowworms * rec_tile * lig_tile  + block_idy * rec_tile * lig_tile  + block_idx ];
    

    double * current_ligand_pose = &d_ligand_pose[block_idz * num_glowworms * ligand_atoms * 4 + block_idy * ligand_atoms * 4];
    double * current_receptor_pose = &d_receptor_pose[block_idz * num_glowworms * receptor_atoms * 4 + block_idy * receptor_atoms * 4];
    // double * current_D_tmp = &D_tmp[block_idz * receptor_atoms * ligand_atoms];
    int moved = d_moved_base[block_idz * num_glowworms + block_idy];
    int steped = d_step_base[block_idz * num_glowworms + block_idy];

    double rho =  d_rho_base[block_idz * num_glowworms + block_idy];
    double luciferins = d_luciferin_base[block_idz * num_glowworms + block_idy];
    double gamma = d_gamma_base[block_idz * num_glowworms + block_idy];
    int reg_dist_to_bins[29] = {
         1,  1,  1,  2,  3,  4,  5,  6,  7,  8,
         9, 10, 11, 12, 13, 14, 14, 15, 15, 16,
        16, 17, 17, 18, 18, 19, 19, 20, 20};
    // double * energy_ptr = &d_energy_tmp[block_idz * num_glowworms * receptor_atoms * ligand_atoms + block_idy * receptor_atoms * ligand_atoms];
    // printf("test \n");
    // double energy = d_scoring_base[block_idz * block_sizey * block_size + block_idy * block_size];
    if((moved || steped == 0)){

    // global memory tensor
    Tensor A = make_tensor(make_gmem_ptr(current_receptor_pose), make_shape(receptor_atoms, 4), make_stride(4, Int<1>{}));
    // B layout 为行优先，tensor行加1stride加K
    Tensor B = make_tensor(make_gmem_ptr(current_ligand_pose), make_shape(ligand_atoms, 4), make_stride(4, Int<1>{}));

    // 获得当前block计算的块的大小 rec和lig 都展开两层循环，使用异步执行, 取出了两个block
    
    Tensor Rec_index = make_tensor(make_gmem_ptr(rec_object), make_shape(receptor_atoms,1), make_stride(Int<1>{}, Int<1>{}));
    Tensor Lig_index = make_tensor(make_gmem_ptr(lig_object), make_shape(ligand_atoms,1),make_stride(Int<1>{}, Int<1>{}));

    // 获得每个block的数据
    Tensor bA = local_tile(A, make_tile(Int<kTileM*2>{}, Int<kTileK>{}), make_coord(block_idx / lig_tile, _));
    Tensor bB = local_tile(B, make_tile(Int<kTileN*2>{}, Int<kTileK>{}), make_coord(block_idx % lig_tile, _));
    Tensor bRec = local_tile(Rec_index, make_tile(Int<kTileM*2>{}, Int<1>{}), make_coord(block_idx / lig_tile, _));
    Tensor bLig = local_tile(Lig_index, make_tile(Int<kTileN*2>{}, Int<1>{}), make_coord(block_idx % lig_tile, _));

    int iter_to_read = 0;
    int iter_to_write = 0;
    int total_read = 4;
    int n_stage = 4;
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    // 一次的的 stage
    // 构造reg 
    // Tensor pipe_A = make_tensor<double>(Shape<_1,_2,_1,_8>{});
    // Tensor pipe_B = make_tensor<double>(Shape<_1,_2,_1,_8>{});
    // Tensor pipe_C = make_tensor<double>(Shape<_2,_2,_2,_8>{});

    

#pragma unroll 
    for(int i = 0; i<2; i++){
// #pragma unroll
//         for(int j = 0; j<2; j++){
            // 获得每个tensorcore计算的数据
            Tensor gA = local_tile(bA(_,_,0), make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(i, _));
            Tensor gB = local_tile(bB(_,_,0), make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(i, _));

            // Tensor grec_idx = local_tile(Rec_index(_,_,0), make_tile(Int<kTileM>{}, Int<1>{}), make_coord(i, _));
            // Tensor glig_idx = local_tile(Lig_index(_,_,0), make_tile(Int<kTileN>{}, Int<1>{}), make_coord(i, _));
            
            // shared memory tensor,获得每个tensorcore对应计算的shmemory
            auto sA = make_tensor(make_smem_ptr(Ashm+i*128),
                                    make_shape(Int<kTileM>{}, Int<kTileK>{},Int<1>{}), make_stride(Int<kTileK>{}, Int<1>{}, Int<1>{}));  
            auto sB = make_tensor(make_smem_ptr(Bshm+i*128),
                                    make_shape(Int<kTileN>{}, Int<kTileK>{},Int<1>{}), make_stride(Int<kTileK>{}, Int<1>{}, Int<1>{})); 
            
            // auto sF = make_tensor(make_smem_ptr(Fshm),
            //                         make_shape(Int<4>{}, Int<kTileN*20>{}, Int<1>{}), make_stride(Int<kTileN*20>{}, Int<1>{}, Int<1>{}));

            
            // 根据mma划分tile
            
            // 对shmem划分到线程
            
            auto tAsA = thr_mma.partition_A(sA);
            auto tBsB = thr_mma.partition_B(sB);


             // 每个线程到共享内存拷贝划分
            G2SCopyA g2s_tiled_copy_a;
            auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(thread_idx);
            auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
            auto tAsA_copy =
            g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

            G2SCopyB g2s_tiled_copy_b;
            auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(thread_idx);
            auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
            auto tBsB_copy =
                g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

             // 异步拷贝
            // gmem-> smem

            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, 0), tAsA_copy(_, _, _, 0));
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, 0), tBsB_copy(_, _, _, 0));
            cp_async_fence();
            ++iter_to_read;
            // cp_async_wait<0>();
            // __syncthreads();
        // }
    }
    // wait one submitted gmem->smem done
    cp_async_wait<1>();

    __syncthreads();

    double thr_energy = 0;
    // Tensor resB1 = make_tensor<double>(Shape<_2,_4>{});
    // Tensor resB2 = make_tensor<double>(Shape<_2,_4>{});
    // Tensor resA1 = make_tensor<double>(Shape<_1,_4>{});
    // Tensor resA2 = make_tensor<double>(Shape<_1,_4>{});
    auto tCrC = make_tensor<double>(Shape<_2,_2,_2>{});


    Tensor r_index = make_tensor<int>(Shape<_1, _2>{});
    Tensor c_index = make_tensor<int>(Shape<_2, _2>{});
    Tensor global_col_idx = make_tensor<int>(Shape<_1,_4>{});
    Tensor global_row_idx = make_tensor<int>(Shape<_1,_2>{});
    Tensor sum_row = make_tensor<double>(Shape<_1,_2>{});
    Tensor sum_col = make_tensor<double>(Shape<_1,_4>{});
#pragma unroll 
    for(int ii = 0; ii<2; ii++){
#pragma unroll
        for(int jj = 0; jj<2; jj++){      


            // 获得每个tensorcore计算的数据
            Tensor gA = local_tile(bA(_,_,0), make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(ii, _));
            Tensor gB = local_tile(bB(_,_,0), make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(jj, _));

            // 获得每一个buffer的index
            Tensor gRec = local_tile(bRec(_,_,0), make_tile(Int<kTileM>{}, Int<1>{}), make_coord(ii, _));
            Tensor gLig = local_tile(bLig(_,_,0), make_tile(Int<kTileN>{}, Int<1>{}), make_coord(jj, _));
            
            // shared memory tensor,获得每个tensorcore对应计算的shmemory
            auto sA = make_tensor(make_smem_ptr(Ashm+ii*128),
                                    make_shape(Int<kTileM>{}, Int<kTileK>{},Int<1>{}), make_stride(Int<kTileK>{}, Int<1>{}, Int<1>{}));  
            auto sB = make_tensor(make_smem_ptr(Bshm+jj*128),
                                    make_shape(Int<kTileN>{}, Int<kTileK>{},Int<1>{}), make_stride(Int<kTileK>{}, Int<1>{}, Int<1>{}));  
            // auto sC = make_tensor(make_smem_ptr(Cshm+(ii*2+jj)*1024),
            //                         make_shape(Int<kTileM>{}, Int<kTileN>{},Int<1>{}), make_stride(Int<kTileN>{}, Int<1>{}, Int<1>{}));

            



        
            // 对32 * 4 的大块划分成 16 * 4 TODO: 此处计算4个tile
            Tensor sA1 = local_tile(sA, make_tile(Int<16>{}, Int<kTileK>{}, Int<1>{}),make_coord(0,_,1));
            Tensor sA2 = local_tile(sA, make_tile(Int<16>{}, Int<kTileK>{}, Int<1>{}),make_coord(1,_,1));

            Tensor iA1 = local_tile(gRec(_,_,0), make_tile(Int<16>{}, Int<1>{}),make_coord(0,_));
            Tensor iA2 = local_tile(gRec(_,_,0), make_tile(Int<16>{}, Int<1>{}),make_coord(1,_));

            Tensor sB1 = local_tile(sB, make_tile(Int<16>{}, Int<kTileK>{}, Int<1>{}),make_coord(0,_,1));
            Tensor sB2 = local_tile(sB, make_tile(Int<16>{}, Int<kTileK>{}, Int<1>{}),make_coord(1,_,1));

            Tensor iB1 = local_tile(gLig(_,_,0), make_tile(Int<16>{}, Int<1>{}),make_coord(0,_));
            Tensor iB2 = local_tile(gLig(_,_,0), make_tile(Int<16>{}, Int<1>{}),make_coord(1,_));

            // if(thread0()){
            //     print_tensor(iA1);
            //     print_tensor(iB1);
            // }
            // 按照warpid继续分块
            // 得到每个warp的8*4小块
            // col
            Tensor warp_col1 = local_tile(sB1(_,_,_,0), make_tile(Int<8>{}, Int<kTileK>{}, Int<1>{}), make_coord(warp_id / 2,_,1));
            Tensor warp_col2 = local_tile(sB2(_,_,_,0), make_tile(Int<8>{}, Int<kTileK>{}, Int<1>{}), make_coord(warp_id / 2,_,1));
            // col
            Tensor warp_idc1 = local_tile(iB1(_,_,0), make_tile(Int<8>{}, Int<1>{}),make_coord(warp_id / 2, _));
            Tensor warp_idc2 = local_tile(iB2(_,_,0), make_tile(Int<8>{}, Int<1>{}),make_coord(warp_id / 2, _));
            // row
            Tensor warp_row1 = local_tile(sA1(_,_,_,0), make_tile(Int<8>{}, Int<kTileK>{}, Int<1>{}), make_coord(warp_id % 2,_,1));
            Tensor warp_row2 = local_tile(sA2(_,_,_,0), make_tile(Int<8>{}, Int<kTileK>{}, Int<1>{}), make_coord(warp_id % 2,_,1));
            // row
            Tensor warp_idr1 = local_tile(iA1(_,_,0), make_tile(Int<8>{}, Int<1>{}), make_coord(warp_id % 2, _));
            Tensor warp_idr2 = local_tile(iA2(_,_,0), make_tile(Int<8>{}, Int<1>{}), make_coord(warp_id % 2, _));

            
            int col_id = (lane_id % 4);
            int row_id = (lane_id / 4);

            // int row_id2 = (thread_idx / 8) + 16;
            Tensor Col_B1 = local_tile(warp_col1(_,_,_,0), make_tile(Int<2>{}, Int<kTileK>{}, Int<1>{}), make_coord(col_id, _, 1));
            Tensor Col_B2 = local_tile(warp_col2(_,_,_,0), make_tile(Int<2>{}, Int<kTileK>{}, Int<1>{}), make_coord(col_id, _, 1));
            
            Tensor Row_A1 = local_tile(warp_row1(_,_,_,0), make_tile(Int<1>{}, Int<kTileK>{}, Int<1>{}), make_coord(row_id, _, 1));
            Tensor Row_A2 = local_tile(warp_row2(_,_,_,0), make_tile(Int<1>{}, Int<kTileK>{}, Int<1>{}), make_coord(row_id,_,1));
            
            Tensor Col_index1 = local_tile(warp_idc1(_,_,0), make_tile(Int<2>{}, Int<1>{}), make_coord(col_id, _));
            Tensor Col_index2 = local_tile(warp_idc2(_,_,0), make_tile(Int<2>{}, Int<1>{}), make_coord(col_id, _));


            Tensor Row_index1 = local_tile(warp_idr1(_,_,0), make_tile(Int<1>{}, Int<1>{}), make_coord(row_id, _));
            Tensor Row_index2 = local_tile(warp_idr2(_,_,0), make_tile(Int<1>{}, Int<1>{}), make_coord(row_id, _));

//             // 对寄存器划分到线程
            auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
            auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
            
            // auto tCrC = thr_mma.partition_fragment_C(sC(_, _, 0));     // (MMA, MMA_M, MMA_N)
            // auto tCsC = thr_mma.partition_C(sC);  // (MMA, MMA_M, MMA_N)
            auto tAsA = thr_mma.partition_A(sA);
            auto tBsB = thr_mma.partition_B(sB);           
            clear(tCrC);


            // double a_value = Ashm[ii * 128 + thread_idx];
            // double b_value = Bshm[ii * 128 + thread_idx];


            //  if(ii == 0 && jj==0 && thread0()){
            //     print_tensor(sA);
            //     print_tensor(sB);
            // }

            // cute::copy(Col_B1, resB1);
            // cute::copy(Col_B2, resB2);
            // cute::copy(Row_A1, resA1);
            // cute::copy(Row_A2, resA2);
            // 拷贝到寄存器中
            // 第一个tile的结果
            cute::copy(Col_index1, c_index(_,0));
            cute::copy(Col_index2, c_index(_,1));
            cute::copy(Row_index1, r_index(_,0));
            cute::copy(Row_index2, r_index(_,1));

            // smem -> reg
            cute::copy(tAsA(_, _, _, 0), tArA);
            cute::copy(tBsB(_, _, _, 0), tBrB);
            // cute::copy(tCrD, tCgD);
            // if(ii == 0 && jj==0 && thread0()){
            //     print_tensor(tArA);
            //     print_tensor(tBrB);
            // }

            cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);

            double v0 = tArA(0); double v1 = tArA(1);
            

            
            
            // reg -> smem
            // double sum_row[2] = {0,0};
            sum_row(0) = 0;
            sum_row(1) = 0;
            double x0, x1, x2, x3;

            x0 = __shfl_sync(FULL_MASK, v0, 0, 4);
            x1 = __shfl_sync(FULL_MASK, v0, 1, 4);
            x2 = __shfl_sync(FULL_MASK, v0, 2, 4);
            x3 = __shfl_sync(FULL_MASK, v0, 3, 4);

            sum_row(0) = ((((x0 * x0) + x1 * x1) + x2 * x2) + x3 * x3);


            // double x0, x1, x2, x3;

            x0 = __shfl_sync(FULL_MASK, v1, 0, 4);
            x1 = __shfl_sync(FULL_MASK, v1, 1, 4);
            x2 = __shfl_sync(FULL_MASK, v1, 2, 4);
            x3 = __shfl_sync(FULL_MASK, v1, 3, 4);

            sum_row(1) = ((((x0 * x0) + x1 * x1) + x2 * x2) + x3 * x3);


            
            
            double v2 = tBrB(0); double v3 = tBrB(1);

            x0 = __shfl_sync(FULL_MASK, v2, 0, 4);
            x1 = __shfl_sync(FULL_MASK, v2, 1, 4);
            x2 = __shfl_sync(FULL_MASK, v2, 2, 4);
            x3 = __shfl_sync(FULL_MASK, v2, 3, 4);
            // use xor
            double sum_0 = ((((x0 * x0) + x1 * x1) + x2 * x2) + x3 * x3);

            // shuffle up4
            double value0 = __shfl_up_sync(0xFFFFFFFF, sum_0, 4,8);
            value0 = __shfl_down_sync(0xFFFFFFFF, value0, 7,16);
            value0 = __shfl_up_sync(0xFFFFFFFF, value0, 4,16);
            value0 = __shfl_up_sync(0xFFFFFFFF, value0, 8,16);
            value0 = __shfl_up_sync(0xFFFFFFFF, value0, 2,4);
            
            value0 = __shfl_down_sync(0xFFFFFFFF, value0, 14,32);
            value0 = __shfl_up_sync(0xFFFFFFFF, value0, 4,8);
            value0 = __shfl_up_sync(0xFFFFFFFF, value0, 8,16);
            value0 = __shfl_up_sync(0xFFFFFFFF, value0, 16,32);
            
            double value1 = __shfl_down_sync(0xFFFFFFFF, sum_0, 4, 8);
            value1 = __shfl_down_sync(0xFFFFFFFF, value1, 7,16);
            value1 = __shfl_up_sync(0xFFFFFFFF, value1, 4,16);
            value1 = __shfl_up_sync(0xFFFFFFFF, value1, 8,16);
            value1 = __shfl_up_sync(0xFFFFFFFF, value1, 2,4);
            
            value1 = __shfl_down_sync(0xFFFFFFFF, value1, 14,32);
            value1 = __shfl_up_sync(0xFFFFFFFF, value1, 4,8);
            value1 = __shfl_up_sync(0xFFFFFFFF, value1, 8,16);
            value1 = __shfl_up_sync(0xFFFFFFFF, value1, 16,32);



            // double x0, x1, x2, x3;

            x0 = __shfl_sync(FULL_MASK, v3, 0, 4);
            x1 = __shfl_sync(FULL_MASK, v3, 1, 4);
            x2 = __shfl_sync(FULL_MASK, v3, 2, 4);
            x3 = __shfl_sync(FULL_MASK, v3, 3, 4);

            double sum_1 = ((((x0 * x0) + x1 * x1) + x2 * x2) + x3 * x3);
            // sum_col(2) = sum_1;

            double value2 = __shfl_up_sync(0xFFFFFFFF, sum_1, 4,8);
            value2 = __shfl_down_sync(0xFFFFFFFF, value2, 7,16);
            value2 = __shfl_up_sync(0xFFFFFFFF, value2, 4,16);
            value2 = __shfl_up_sync(0xFFFFFFFF, value2, 8,16);
            value2 = __shfl_up_sync(0xFFFFFFFF, value2, 2,4);
            
            value2 = __shfl_down_sync(0xFFFFFFFF, value2, 14,32);
            value2 = __shfl_up_sync(0xFFFFFFFF, value2, 4,8);
            value2 = __shfl_up_sync(0xFFFFFFFF, value2, 8,16);
            value2 = __shfl_up_sync(0xFFFFFFFF, value2, 16,32);
            
            double value3 = __shfl_down_sync(0xFFFFFFFF, sum_1, 4, 8);
            value3 = __shfl_down_sync(0xFFFFFFFF, value3, 7,16);
            value3 = __shfl_up_sync(0xFFFFFFFF, value3, 4,16);
            value3 = __shfl_up_sync(0xFFFFFFFF, value3, 8,16);
            value3 = __shfl_up_sync(0xFFFFFFFF, value3, 2,4);
            
            value3 = __shfl_down_sync(0xFFFFFFFF, value3, 14,32);
            value3 = __shfl_up_sync(0xFFFFFFFF, value3, 4,8);
            value3 = __shfl_up_sync(0xFFFFFFFF, value3, 8,16);
            value3 = __shfl_up_sync(0xFFFFFFFF, value3, 16,32);

            sum_col(0) = value0;
            sum_col(1) = value1;
            sum_col(2) = value2;
            sum_col(3) = value3;


            // int current_num[6]={0,0,0,0,0,0};

            
            // 全局坐标索引
            // col
            global_col_idx(0,0) = (block_idx % lig_tile) * kTileN * 2 + jj * kTileN + 0 * 16 + warp_id / 2 * 8 + col_id * 2;
            global_col_idx(0,1) = (block_idx % lig_tile) * kTileN * 2 + jj * kTileN + 0 * 16 + warp_id / 2 * 8 + col_id * 2 + 1;

            global_col_idx(0,2) = (block_idx % lig_tile) * kTileN * 2 + jj * kTileN + 1 * 16 + warp_id / 2 * 8 + col_id * 2;
            global_col_idx(0,3) = (block_idx % lig_tile) * kTileN * 2 + jj * kTileN + 1 * 16 + warp_id / 2 * 8 + col_id * 2 + 1;



            global_row_idx(0,0) = (block_idx / lig_tile) * kTileM * 2 + ii * kTileM + 0 * 16 + warp_id % 2 * 8 + row_id;
            global_row_idx(0,1) = (block_idx / lig_tile) * kTileM * 2 + ii * kTileM + 1 * 16 + warp_id % 2 * 8 + row_id;
            for(int i = 0; i<2; i++){
                r_index(0,i) = r_index(0,i) * 168 * 20;
            }
            // add x2
            for(int k = 0; k<2; k++){
                for(int i = 0; i<2; i++){
                // row
                    for(int j = 0; j<2; j++){
                    // rep
                        double tmp;
                        if(global_col_idx(0,k*2+i)>=lig_real_len || global_row_idx(0,j)>=rec_real_len){
                            tCrC(i,j,k) = 0;
                        }else{
                            tCrC(i,j,k) = tCrC(i,j,k) * -2 + sum_row(j) + sum_col(k*2+i);
                        }
                        if(tCrC(i,j,k) <= 225 && tCrC(i,j,k) > 0){
                            int d = (sqrt(tCrC(i,j,k))*2.0 - 1.0);
                            int dfire_bin = reg_dist_to_bins[d] - 1;
                            int atomb = c_index(i,k) * 20;
                            unsigned int array = r_index(0,j) + atomb + dfire_bin;
                            // tCrC(i,j,k) = (double)(array);
                            tmp = d_fastdfire[array];

                        }else{
                            tmp = 0;
                        }
                        thr_energy += tmp;
                        
                    }
                }
            }
            
        }
    }
    
    
    Rshm[thread_idx] = thr_energy;
    __syncthreads();
    for(int offset = block_size >> 1; offset >= 32; offset >>=1){
        if(thread_idx < offset){
            Rshm[thread_idx] += Rshm[thread_idx+offset];
        }
        __syncthreads();
    }
    double y = Rshm[thread_idx];
    
    for(int offset = 16; offset > 0; offset>>=1){
        y+=__shfl_down_sync(FULL_MASK,y,offset);
    }
    if(thread_idx ==0){
        current_reduce[thread_idx] = y;
    }
    
    }

}


template<typename Key,
    int         BLOCK_THREADS,
    int         ITEMS_PER_THREAD>
__launch_bounds__(BLOCK_THREADS)
__global__ void EnerSortKernel(
    Key * d_in, Key * d_out, 
    int * d_moved_base, int * d_step_base,
    int rec_tile, int lig_tile
){

    // int n = blockIdx.x
    int thread_idx = threadIdx.x;
    int block_idx = blockIdx.x;
    int moved = d_moved_base[block_idx];
    int steped = d_step_base[block_idx];
    if(moved || steped == 0){
        // using namespace cute;
        // enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
        // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
        // typedef cub::BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
        // Specialize BlockRadixSort type for our thread block
        typedef cub::BlockRadixSort<uint64_t, BLOCK_THREADS, 1> BlockRadixSortT;

        // __shared__ Key s_mem[3000];
        __shared__ typename BlockRadixSortT::TempStorage   tmp_sort;
        int length = rec_tile * lig_tile;
        Key * current_din = &d_in[block_idx * length];
        Key * current_dout = &d_out[block_idx * length];
        // int gmem_idx = thread_idx * ITEMS_PER_THREAD;
        // share memory 规约

        const int stride  = blockDim.x ;
        double y[1] = {0};
        uint64_t item_uint[1] = {0};
#pragma unroll
        for(int n = thread_idx ; n<length; n += stride){
            y[0] += current_din[n];
        }
        item_uint[0] = *reinterpret_cast<uint64_t*>(&y[0]);
        BlockRadixSortT(tmp_sort).SortDescending(item_uint);
        y[0] = *reinterpret_cast<double*>(&item_uint);
        // s_mem[thread_idx] = y;
        // __syncthreads();

        current_dout[thread_idx] = y[0];
    }
    


}


__global__ void gnerate_energy(
    double * d_scoring_base, double * d_luciferin_base, double * d_rho_base, double * d_gamma_base,
    double * sorted_energy, int * d_moved_base, int * d_step_base,
    int swarms, int num_glowworms, int anm_lig, int anm_rec,
    int rec_tile, int lig_tile , int length
){
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int block_size = blockDim.x;
    int moved = d_moved_base[block_idx*num_glowworms + thread_idx];
    int steped = d_step_base[block_idx*num_glowworms + thread_idx];
    double rho = d_rho_base[block_idx * num_glowworms + thread_idx];
    double luciferins = d_luciferin_base[block_idx*num_glowworms+thread_idx];
    double gamma = d_gamma_base[block_idx*num_glowworms+thread_idx];
    if(moved || steped == 0){
        

        
        double * current_energy_tmp = &sorted_energy[block_idx * num_glowworms * rec_tile * lig_tile + thread_idx * rec_tile * lig_tile];
        double ener_sum = 0;
        // __shared__ double s_mem[2048];
        // s_mem[thread_idx] = 0;
#pragma unroll  
        for(int i = 0; i < rec_tile * lig_tile; i++){

            ener_sum += current_energy_tmp[i];
        }
        ener_sum = ((ener_sum)*0.0157 - 4.7)*-1;
        // printf("%f \n ", ener_sum);
        d_scoring_base[block_idx * num_glowworms + thread_idx] = ener_sum;
    }
    d_luciferin_base[block_idx * num_glowworms + thread_idx] = (1.0 - rho) * d_luciferin_base[block_idx * num_glowworms + thread_idx] + gamma * d_scoring_base[block_idx * num_glowworms + thread_idx];
    

}

__global__ void generate_energy_v2(
    double * d_scoring_base, double * d_luciferin_base, double * d_rho_base, double * d_gamma_base,
    double * sorted_energy, int * d_moved_base, int * d_step_base,
    size_t swarms, size_t num_glowworms, size_t anm_lig, size_t anm_rec,
    size_t rec_tile, size_t lig_tile , size_t length
){
    

    size_t block_idy = blockIdx.y;
    size_t block_idx = blockIdx.x;

    size_t thread_idx = threadIdx.x;
    size_t block_size = blockDim.x;

    size_t stride = block_size;


    __shared__ double s_y[1024];
    int moved = d_moved_base[block_idy*num_glowworms + block_idx];
    int steped = d_step_base[block_idy*num_glowworms + block_idx];
    double rho = d_rho_base[block_idy * num_glowworms + block_idx];
    double luciferins = d_luciferin_base[block_idy*num_glowworms+block_idx];
    double gamma = d_gamma_base[block_idy*num_glowworms+block_idx];
    if(moved || steped == 0){
        double * current_energy_tmp = &sorted_energy[block_idy * num_glowworms * rec_tile * lig_tile + block_idx * rec_tile * lig_tile];
        double y = 0;
        for(int n = thread_idx; n< rec_tile * lig_tile; n+=stride){
            y += current_energy_tmp[n];
        }
        s_y[thread_idx] = y;
        __syncthreads();
        for(int offset = block_size >> 1; offset >= 32; offset >>= 1){
            if (thread_idx < offset)
            {
                s_y[thread_idx] += s_y[thread_idx + offset];
            }
            __syncthreads();
        }
        y = s_y[thread_idx];
        thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
        for (int i = g.size() >> 1; i > 0; i >>= 1)
        {
            y += g.shfl_down(y, i);
        }
        if(thread_idx==0){
            y = ((y)*0.0157 - 4.7)*-1;
            d_scoring_base[block_idy * num_glowworms + block_idx] = y;

        }
        
    }
    if(thread_idx==0){
        d_luciferin_base[block_idy * num_glowworms + block_idx] = (1.0 - rho) * d_luciferin_base[block_idy * num_glowworms + block_idx] + gamma * d_scoring_base[block_idy * num_glowworms + block_idx];
    
    }

}




static size_t get_chunk_memory(SwarmCenters & centers,Complex & receptor, Complex &lignad, int swarms, int num_glowworms, int object_size_rec, int object_size_lig, int rec_len, int lig_len, int rec_tile, int lig_tile){
    size_t total_memory = sizeof(double) * swarms*num_glowworms*rec_len*4; // rec pose
    total_memory += sizeof(double) * swarms*num_glowworms*lig_len*4; // lig pose
    total_memory += sizeof(double) * swarms*num_glowworms*3; //lig ref
    total_memory += sizeof(double) * swarms * num_glowworms * 3; // delta
    total_memory += sizeof(double) * swarms*centers.pos_len; // center pos
    total_memory += sizeof(double)*swarms*centers.pos_len; // postion pos
    total_memory += sizeof(int) * swarms * num_glowworms; // id
    total_memory += sizeof(double) * swarms * num_glowworms; // score
    total_memory += sizeof(double) * swarms * num_glowworms; // luciferin
    total_memory += sizeof(double) * swarms * num_glowworms; //rho
    total_memory += sizeof(double) * swarms * num_glowworms; //gamma
    total_memory += sizeof(double) * swarms * num_glowworms; // beta
    total_memory += sizeof(double) * swarms * num_glowworms; // vision
    total_memory += sizeof(double) * swarms * num_glowworms; // max vision
    total_memory += sizeof(int) * swarms * num_glowworms; // step
    total_memory += sizeof(int) * swarms * num_glowworms; //moved
    total_memory += sizeof(int) * swarms * num_glowworms; //max neighbor
    total_memory += sizeof(int) * swarms * num_glowworms; // nnei_len
    total_memory += sizeof(int)*swarms*num_glowworms*num_glowworms; // neighbor
    total_memory += sizeof(double) * STEPS * swarms * num_glowworms; //prob
    total_memory += sizeof(unsigned int) * 50;   // dist to bins
    total_memory += sizeof(int)*swarms*num_glowworms; //select
    total_memory += sizeof(double)*swarms*num_glowworms*num_glowworms; //prob
    total_memory += sizeof(double)*swarms*num_glowworms*centers.anm_rec; //rec_base
    total_memory += sizeof(double)*swarms*num_glowworms*centers.anm_lig; // lig_base
    total_memory += sizeof(double)*receptor.num_atoms*3; // receptor
    total_memory += sizeof(double)*lignad.num_atoms*3; // ligand
    total_memory += sizeof(double)*3; //lig ref
    total_memory += sizeof(double)*DEFAULT_NMODES_REC*receptor.num_atoms*3; // rec modes
    total_memory += sizeof(double)*DEFAULT_NMODES_LIG*lignad.num_atoms*3; // lig modes
    total_memory += sizeof(int)*receptor.num_atoms; // rec mask
    total_memory += sizeof(int)*lignad.num_atoms; // lig mask
    total_memory += sizeof(double)*168*168*20; // difire
    total_memory += sizeof(int)*object_size_rec; //rec object
    total_memory += sizeof(int)*object_size_lig; // lig object
    total_memory += sizeof(double)*swarms * num_glowworms * rec_tile * lig_tile; //ener
    return total_memory;
}


void cal_gso_tasks_gpu(SwarmCenters & centers,Complex & receptor, Complex &lignad, FastDifire & fastdifire, int seed, double step_translation, double step_rotation, bool use_anm, double nmodes_step,
    int anm_rec, int anm_lig, bool local_minimization, int swarms, int num_glowworms, double * receptor_reference_points, double * receptor_poles, double * ligand_reference_points, double * ligand_poles,
    int *dfire_objects_rec, int *dfire_objects_lig,
    int object_size_rec, int object_size_lig, int rank, int size)
{   
    // int chunks = get_chunks();
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int gpu_device = rank % deviceCount;
    cudaSetDevice(gpu_device);
    // std::cout<<rank<<":"<<gpu_device<<":"<<deviceCount<<std::endl;
    static int num_threads = get_env_num_threads();
    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    // if(rank == 0){
    //     std::cout << "Total memory: " << totalMem / (1024 * 1024) << " MB" << std::endl;
    //     std::cout << "Free memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;
    // }
    // 
    // std::cout<<sizeof(size_t)<<"  test "<<std::endl;

    // int num_threads = get_env_num_threads();
    DockingInterface rec_interface, lig_interface;

    config::GemmConfig<32,32,4> gemm_config;
    constexpr int kTileM = 32; 
    constexpr int kTileN = 32; 
    constexpr int kTileK = 4; 
    dim3 block_tensorcore(gemm_config.block);
    // pad 
    int rec_grid = (receptor.num_atoms + kTileM - 1) / kTileM; 
    int lig_grid = (lignad.num_atoms + kTileN - 1) / kTileN;
    int rec_len = rec_grid * kTileM;
    int lig_len = lig_grid * kTileN;
    int rec_tile = rec_len/(kTileM*2);
    int lig_tile = lig_len/(kTileN*2);
    
    
    // initial
    int swarm_len = centers.swarm_chunk;
    int chunks = 1;
    size_t total_required_memory = get_chunk_memory(centers,receptor, lignad, swarm_len,num_glowworms,object_size_rec,object_size_lig, rec_len, lig_len, rec_tile, lig_tile);
    if(rank == 0){
        std::cout<<"total memory: "<<total_required_memory<<std::endl;
    }
    
    // size_t total_memory = get_chunk_memory(centers,receptor, lignad, swarm_len,num_glowworms,object_size_rec,object_size_lig, rec_len, lig_len, rec_tile, lig_tile);
    // auto chunking
    // unsigned long current_mem = get_chunk_memory(centers,receptor, lignad, swarm_len,num_glowworms,object_size_rec,object_size_lig, rec_len, lig_len, rec_tile, lig_tile);
    // freeMem = freeMem;
    // two stream
    int stream = 2;
    // if(get_chunk_memory(centers,receptor, lignad, swarm_len,num_glowworms,object_size_rec,object_size_lig, rec_len, lig_len, rec_tile, lig_tile) >= freeMem){
    //     // std::cout<<"large"<<std::endl;
    // } else{
    //     std::cout<<get_chunk_memory(centers,receptor, lignad, centers.swarm_chunk, num_glowworms, object_size_rec, object_size_lig, rec_len, lig_len, rec_tile, lig_tile)<<std::endl;
    //     // std::cout<<freeMem<<std::endl;
    // }
    while(get_chunk_memory(centers,receptor, lignad, swarm_len,num_glowworms,object_size_rec,object_size_lig, rec_len, lig_len, rec_tile, lig_tile) >= freeMem){
        // std::cout<<"current out of memory"<<chunks<<":"<<swarm_len<<std::endl;
        chunks ++ ;
        swarm_len = (centers.swarm_chunk + chunks - 1)  / chunks;
        
    }
    
    // std::cout << "swarms" <<swarms<< std::endl;
    // std::cout<<current_mem<<":"<<freeMem<<std::endl;

    // int swarm_chunks = (swarms + chunks - 1)  / chunks;
    // 申请的swarm大小
    // int swarm_len = (chunks > 1) ? swarm_chunks:swarms;



    int *id_base;
    CHECK(cudaMallocHost((void **)&id_base, sizeof(int) * swarm_len * num_glowworms));
    double *scoring_base;
    CHECK(cudaMallocHost((void **)&scoring_base, sizeof(double) * swarm_len * num_glowworms));

    double *luciferin_base;
    CHECK(cudaMallocHost((void **)&luciferin_base, sizeof(double) * swarm_len * num_glowworms));

    double *rho_base;
    CHECK(cudaMallocHost((void **)&rho_base, sizeof(double) * swarm_len * num_glowworms));

    double *gamma_base;
    CHECK(cudaMallocHost((void **)&gamma_base, sizeof(double) * swarm_len * num_glowworms));

    double *beta_base;
    CHECK(cudaMallocHost((void **)&beta_base, sizeof(double) * swarm_len * num_glowworms));

    double *vision_range_base;
    CHECK(cudaMallocHost((void **)&vision_range_base, sizeof(double) * swarm_len * num_glowworms));

    double *max_vision_range_base;
    CHECK(cudaMallocHost((void **)&max_vision_range_base, sizeof(double) * swarm_len * num_glowworms));

    int *step_base;
    CHECK(cudaMallocHost((void **)&step_base, sizeof(int) * swarm_len * num_glowworms));

    int *moved_base;
    CHECK(cudaMallocHost((void **)&moved_base, sizeof(int) * swarm_len * num_glowworms));

    int *max_neighbors_base;
    CHECK(cudaMallocHost((void **)&max_neighbors_base, sizeof(int) * swarm_len * num_glowworms));

    int *nnei_len_base;
    CHECK(cudaMallocHost((void **)&nnei_len_base, sizeof(int) * swarm_len * num_glowworms));

    double *ligand_reference_pose;
    CHECK(cudaMallocHost((void **)&ligand_reference_pose, sizeof(double) * swarm_len * num_glowworms * 3));

    int *select_base;
    CHECK(cudaMallocHost((void **)&select_base, sizeof(int) * swarm_len * num_glowworms));

    double *delta_base;
    CHECK(cudaMallocHost((void **)&delta_base, sizeof(double) * swarm_len * 3));

    double *prob_array;
    CHECK(cudaMallocHost((void **)&prob_array, sizeof(double) * STEPS * swarm_len * num_glowworms));


    

    

    double * d_receptor_pose,* d_ener, * d_ligand_pose, * d_ligand_reference_pose ,  *d_select_position_base, *d_probabilities, *d_energy, * d_energy_tmp, *d_dist,
    * d_delta_base,*d_scoring_base, *d_luciferin_base, *d_rho_base, *d_gamma_base, *d_beta_base, *d_vision_range_base, *d_max_vision_range_base, 
    *d_prob_array, * d_centers_pos, * d_receptor_atom_coordinates, *d_ligand_atom_coordinates, *d_current_ligand_reference_pose, *d_receptor_modes, *d_ligand_modes,
    *rotate_inverse, * current_pos, * current_pos2, *d_fastdfire, * d_rec_base, * d_lig_base;
    int *d_id_base, *d_step_base, *d_moved_base, *d_max_neighbors_base, *d_nnei_len_base, * d_selece_base,
    *d_neighbors_base, * d_receptor_mask, *d_ligand_mask, *d_dfire_objects_rec, *d_dfire_objects_lig;
    unsigned int * indexes, *array_index, * d_dist_to_bins;

    CHECK(cudaMalloc((void**)&d_dist_to_bins,sizeof(unsigned int) * 50));
    CHECK(cudaMalloc((void **)& d_receptor_atom_coordinates, sizeof(double)*receptor.num_atoms*3));
    CHECK(cudaMalloc((void **)& d_ligand_atom_coordinates, sizeof(double)*lignad.num_atoms*3));
    CHECK(cudaMalloc((void **)& d_current_ligand_reference_pose, sizeof(double)*3));
    CHECK(cudaMalloc((void **)& d_receptor_modes, sizeof(double)*DEFAULT_NMODES_REC*receptor.num_atoms*3));
    CHECK(cudaMalloc((void **)& d_ligand_modes, sizeof(double)*DEFAULT_NMODES_LIG*lignad.num_atoms*3));
    CHECK(cudaMalloc((void **)& d_receptor_mask, sizeof(int)*receptor.num_atoms));
    CHECK(cudaMalloc((void **)& d_ligand_mask, sizeof(int)*lignad.num_atoms));
    CHECK(cudaMalloc((void **)& d_fastdfire, sizeof(double)*168*168*20));
    CHECK(cudaMalloc((void **)& d_dfire_objects_rec,sizeof(int)*object_size_rec));
    CHECK(cudaMalloc((void **)& d_dfire_objects_lig,sizeof(int)*object_size_lig));

    CHECK(cudaMemcpy(d_dist_to_bins, dist_to_bins,sizeof(unsigned int) * 50, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_receptor_atom_coordinates, receptor.atom_coordinates, sizeof(double)*receptor.num_atoms*3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ligand_atom_coordinates, lignad.atom_coordinates, sizeof(double)*lignad.num_atoms*3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_current_ligand_reference_pose, ligand_reference_points, sizeof(double)*3,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_receptor_modes, receptor.modes, sizeof(double)* DEFAULT_NMODES_REC * receptor.num_atoms*3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ligand_modes, lignad.modes, sizeof(double) * DEFAULT_NMODES_LIG * lignad.num_atoms*3, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_receptor_mask, receptor.mask,sizeof(int)*receptor.num_atoms,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ligand_mask, lignad.mask,sizeof(int)*lignad.num_atoms,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_fastdfire, fastdifire.difire_energy, sizeof(double)*168*168*20,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dfire_objects_rec,dfire_objects_rec,sizeof(int)*object_size_rec,cudaMemcpyHostToDevice ));
    CHECK(cudaMemcpy(d_dfire_objects_lig,dfire_objects_lig,sizeof(int)*object_size_lig,cudaMemcpyHostToDevice));



    // std::cout<<rec_len<<":"<<lig_len<<std::endl;
    CHECK(cudaMalloc((void **) &d_receptor_pose,sizeof(double) * swarm_len*num_glowworms*rec_len*4));
    CHECK(cudaMalloc((void **) &d_ligand_pose,sizeof(double) * swarm_len*num_glowworms*lig_len*4));
    CHECK(cudaMalloc((void **) &d_ligand_reference_pose,sizeof(double) * swarm_len*num_glowworms*3));
    CHECK(cudaMalloc((void **) &d_delta_base,sizeof(double) * swarm_len * num_glowworms * 3));
    CHECK(cudaMalloc((void **) &d_centers_pos, sizeof(double) * swarm_len*centers.pos_len));
    CHECK(cudaMalloc((void **) &d_select_position_base, sizeof(double)*swarm_len*centers.pos_len));
    CHECK(cudaMalloc((void**)&d_id_base, sizeof(int) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_scoring_base, sizeof(double) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_luciferin_base, sizeof(double) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_rho_base, sizeof(double) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_gamma_base, sizeof(double) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_beta_base, sizeof(double) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_vision_range_base, sizeof(double) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_max_vision_range_base, sizeof(double) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_step_base, sizeof(int) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_moved_base, sizeof(int) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_max_neighbors_base, sizeof(int) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_nnei_len_base, sizeof(int) * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_neighbors_base, sizeof(int) * swarm_len * num_glowworms * num_glowworms));
    CHECK(cudaMalloc((void**)&d_prob_array, sizeof(double) * STEPS * swarm_len * num_glowworms));
    CHECK(cudaMalloc((void**)&d_selece_base,sizeof(int)*swarm_len*num_glowworms));
    CHECK(cudaMalloc((void**)&d_probabilities,sizeof(double)*swarm_len*num_glowworms*num_glowworms));
    CHECK(cudaMalloc((void **)&d_rec_base, sizeof(double)*swarm_len*num_glowworms*centers.anm_rec));
    CHECK(cudaMalloc((void **)&d_lig_base, sizeof(double)*swarm_len*num_glowworms*centers.anm_lig));
    CHECK(cudaMalloc((void **) & d_ener, sizeof(double)*swarm_len * num_glowworms * rec_tile * lig_tile ));


    

    int current_chunks = 0;
    // from swarms_start to +swarm_chunk
    for(int swarms_start = centers.swarm_start; swarms_start < centers.swarm_start+centers.swarm_chunk; swarms_start += swarm_len){
        
    int c_swarm_len = (swarms_start + swarm_len > swarms) ? swarms-swarms_start : swarm_len;

    int block_size = 256;

    int grid_x_r = (receptor.num_atoms + block_size - 1) / block_size;
    int grid_x_l = (lignad.num_atoms + block_size - 1) / block_size;
    int grid_y = num_glowworms;
    int grid_z = c_swarm_len;
    // dim3 block(block_size, 1, 1);
    dim3 grid_r(grid_x_r,grid_y,grid_z);
    dim3 grid_l(grid_x_l,grid_y,grid_z);
    dim3 grid_score(grid_y,grid_z,1);

    dim3 grid_tensorcore(rec_tile * lig_tile , num_glowworms, c_swarm_len);
    #pragma omp parallel for
    for (int i = 0; i < c_swarm_len*num_glowworms; i++)
    {
        id_base[i] = 0;
        scoring_base[i] = 0;
        luciferin_base[i] = 5.0;
        rho_base[i] = 0.4;
        gamma_base[i] = 0.6;
        beta_base[i] = 0.08;
        vision_range_base[i] = 0.2;
        max_vision_range_base[i] = 5.0;
        step_base[i] = 0;
        moved_base[i] = 0;
        max_neighbors_base[i] = 5;
        nnei_len_base[i] = 0;
    }
    
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 gen(0+swarms_start);
    #pragma omp parallel for
    for(int i = 0; i<c_swarm_len; i++){
        for(int g = 0; g<num_glowworms; g++){
            id_base[i*num_glowworms+g] = g;
        }
    }
    // 构造 random array
    #pragma omp parallel for
    for (int i = 0; i< c_swarm_len; i++){
        for(int s = 0; s<STEPS; s++){
            for(int j = 0; j<num_glowworms; j++ ){
                double value  = distribution(gen);
                prob_array[i*STEPS*num_glowworms + s*num_glowworms + j] = value;
            }
        }
    }
    // memcpy
    // std::cout<<swarms_start<<":"<<c_swarm_len<<":"<<swarms<<std::endl;
    CHECK(cudaMemcpy(d_centers_pos, centers.pos + swarms_start * centers.pos_len, sizeof(double)*c_swarm_len* centers.pos_len,cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_receptor_pose,0, sizeof(double) * c_swarm_len*num_glowworms*rec_len*4));
    CHECK(cudaMemset(d_ligand_pose, 0, sizeof(double) * c_swarm_len*num_glowworms*lig_len*4));
    CHECK(cudaMemcpy(d_id_base, id_base, sizeof(int) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scoring_base, scoring_base, sizeof(double) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_luciferin_base, luciferin_base, sizeof(double) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_rho_base, rho_base, sizeof(double) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_gamma_base, gamma_base, sizeof(double) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_beta_base, beta_base, sizeof(double) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vision_range_base, vision_range_base, sizeof(double) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_max_vision_range_base, max_vision_range_base, sizeof(double) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_step_base, step_base, sizeof(int) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_moved_base, moved_base, sizeof(int) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_max_neighbors_base, max_neighbors_base, sizeof(int) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_nnei_len_base, nnei_len_base, sizeof(int) * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_prob_array, prob_array, sizeof(double) * STEPS * c_swarm_len * num_glowworms, cudaMemcpyHostToDevice));

    // float prepare_receptor, prepare_ligand, cal_dfire, t_score, cal_neighbor, cal_move, write_centerpos, t_pdb, cluster;
    if(rank == 0){
        std::cout<<"prepare docking"<<std::endl;
    }
    for(int step = 0; step<STEPS; step++){
        // std::cout<<"proc:"<<rank<<"; chunk:"<<current_chunks<<"; step:"<<step<<std::endl;

        
        prepare_receptor_pose<<<grid_r,block_size>>>(
            d_centers_pos, d_receptor_pose, 
            d_receptor_atom_coordinates, 
            d_receptor_modes, 
            d_receptor_mask,
            d_moved_base, d_step_base,
            c_swarm_len, num_glowworms,  centers.anm_lig, centers.anm_rec,
            receptor.num_atoms, rec_len
        );
    

        prepare_lignad_pose<<<grid_l,block_size>>>(
            d_centers_pos,  d_ligand_pose,  d_ligand_reference_pose, 
            d_ligand_atom_coordinates,  
            d_current_ligand_reference_pose, 
            d_ligand_modes,
            d_ligand_mask,
            d_moved_base, d_step_base,
            c_swarm_len, num_glowworms,  centers.anm_lig, centers.anm_rec,
            lignad.num_atoms, lig_len
        );
        
        
        #if defined(GPU_4090)
        // std::cout<<4090<<std::endl;
        calculate_dfire_tensorcore_cute2<decltype(gemm_config),128,8><<<grid_tensorcore, 128>>>(
            d_ener,
            d_receptor_pose, d_ligand_pose,
            d_scoring_base, d_luciferin_base, d_rho_base, d_gamma_base,
            d_moved_base, d_step_base,
            d_fastdfire, d_dfire_objects_rec, d_dfire_objects_lig,d_dist_to_bins,
            c_swarm_len, num_glowworms, 
            rec_len, lig_len, rec_tile, lig_tile, receptor.num_atoms, lignad.num_atoms
        );
        #else
        calculate_dfire_tensorcore_cute3<decltype(gemm_config),128,8><<<grid_tensorcore, 128>>>(
            d_ener,
            d_receptor_pose, d_ligand_pose,
            d_scoring_base, d_luciferin_base, d_rho_base, d_gamma_base,
            d_moved_base, d_step_base,
            d_fastdfire, d_dfire_objects_rec, d_dfire_objects_lig,d_dist_to_bins,
            c_swarm_len, num_glowworms, 
            rec_len, lig_len, rec_tile, lig_tile, receptor.num_atoms, lignad.num_atoms
        );
        #endif


        generate_energy_v2<<<c_swarm_len, num_glowworms>>>(
            d_scoring_base, d_luciferin_base, d_rho_base, d_gamma_base,
            d_ener, d_moved_base, d_step_base,
            c_swarm_len, num_glowworms, anm_lig, anm_rec,
            rec_tile, lig_tile , 512
        );
        

        cal_move_neighbors<<<c_swarm_len, num_glowworms>>>(
            d_select_position_base, d_neighbors_base, d_vision_range_base, d_selece_base, d_nnei_len_base,
            d_luciferin_base, d_probabilities, d_prob_array, d_centers_pos, d_ligand_reference_pose,
            c_swarm_len, num_glowworms, anm_lig, anm_rec,
            lignad.num_atoms,step
        );
        
        
        move_position_v2<<<c_swarm_len,num_glowworms>>>(
        d_vision_range_base, d_centers_pos, d_select_position_base, 
        d_id_base, d_selece_base, d_delta_base, d_rec_base, d_lig_base,
        d_beta_base, d_max_neighbors_base,
        d_max_vision_range_base, d_nnei_len_base,
        d_moved_base, c_swarm_len, num_glowworms,anm_lig,  anm_rec,
         lignad.num_atoms,step
        );
        cudaDeviceSynchronize();
        // exit(0);
    }

    cudaDeviceSynchronize();
    
    CHECK(cudaMemcpy(centers.pos + swarms_start * centers.pos_len,d_centers_pos, sizeof(double)*c_swarm_len* centers.pos_len,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(luciferin_base,d_luciferin_base, sizeof(double)*c_swarm_len* num_glowworms,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(nnei_len_base,d_nnei_len_base,sizeof(int)*c_swarm_len* num_glowworms,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(vision_range_base ,d_vision_range_base, sizeof(double)*c_swarm_len* num_glowworms,cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(scoring_base,d_scoring_base, sizeof(double)*c_swarm_len* num_glowworms,cudaMemcpyDeviceToHost));

    // std::cout<<"finished"<<rank<<std::endl;
    // exit(0);
    if(rank == 0){
        std::cout<<"saving pos file"<<std::endl;
    }
    // std::cout<<"saving:"<<" rank: "<<rank<<" swarm_start:"<<swarms_start<<" swarm_end:"<<swarms_start+c_swarm_len<<" swarm_len:"<<c_swarm_len<<std::endl;
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<c_swarm_len; i++){


        double *current_pos = &centers.pos[swarms_start * centers.pos_len + i*centers.pos_len];
        

        int select_index = 0;

        double * delta = &delta_base[i*3];


        string save_swarm = "swarm_";
        string file_name = "gso_";
        string swarms = to_string(i + swarms_start);
        string steps = to_string(STEPS) ;
        string file_type = ".out";
        string save_dir =  save_swarm+swarms;
        string save_file = file_name+steps+file_type;
        // std::cout<<save_file<<std::endl;
        string slash = "/";
        string full_path = save_dir+slash+save_file;
        int fd = open(full_path.c_str(), O_RDWR | O_CREAT, 0666);
        if(fd == -1){
            perror("open");
            exit(1);
        }
        
        // std::ofstream newFile(full_path.c_str());
        
        string file_buffer = "#Coordinates  RecID  LigID  Luciferin  Neighbor's number  Vision Range  Scoring\n";
        
        for(int g = 0; g<num_glowworms;g++){
            string line = "(";
            double * current_postion = &current_pos[g*(7+centers.anm_rec+centers.anm_lig)];
            for(int pp = 0; pp < (7+centers.anm_rec+centers.anm_lig); pp++){
                line = line + to_string(current_postion[pp]);
                if(pp<(7+centers.anm_rec+centers.anm_lig)-1){
                    line = line + ", ";
                }
                
            }
            line = line +")\t";
            line = line + to_string(0) + "\t";
            line = line + to_string(0) + "\t";
            line = line + to_string(luciferin_base[i*num_glowworms+g]) + "\t";
            line = line + to_string(nnei_len_base[i*num_glowworms+g]) + "\t";
            line = line + to_string(vision_range_base[i*num_glowworms+g]) + "\t";
            line = line + to_string(scoring_base[i*num_glowworms+g]) + "\t";
            line = line +"\n";
            file_buffer  = file_buffer + line;
            // newFile<<"\n";
        }
        int str_len = strlen(file_buffer.c_str());
        // std::cout<<str_len<<std::endl;
        lseek(fd,str_len-1,SEEK_END);  
        write(fd, "", 1);

        char *p = (char *)mmap(NULL, str_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }
        
        memcpy(p, file_buffer.c_str(),str_len);
        munmap(p, str_len);
        close(fd);
    }

    }

    

    cudaFreeHost(id_base);
    cudaFreeHost(scoring_base);
    cudaFreeHost(luciferin_base);
    cudaFreeHost(rho_base);
    cudaFreeHost(gamma_base);
    cudaFreeHost(beta_base);
    cudaFreeHost(vision_range_base);
    cudaFreeHost(max_vision_range_base);
    cudaFreeHost(step_base);
    cudaFreeHost(moved_base);
    cudaFreeHost(max_neighbors_base);
    cudaFreeHost(nnei_len_base);
    cudaFreeHost(ligand_reference_pose);
    cudaFreeHost(select_base);
    cudaFreeHost(delta_base);
    cudaFreeHost(prob_array);


    // freeMemoryPool(&mem_pool);

    CHECK(cudaFree(d_receptor_pose));
    CHECK(cudaFree(d_ligand_pose));
    CHECK(cudaFree(d_ligand_reference_pose));
    CHECK(cudaFree(d_delta_base));
    CHECK(cudaFree(d_centers_pos));
    CHECK(cudaFree(d_select_position_base));
    CHECK(cudaFree(d_id_base));
    CHECK(cudaFree(d_scoring_base));
    CHECK(cudaFree(d_luciferin_base));
    CHECK(cudaFree(d_rho_base));
    CHECK(cudaFree(d_gamma_base));
    CHECK(cudaFree(d_beta_base));
    CHECK(cudaFree(d_vision_range_base));
    CHECK(cudaFree(d_max_vision_range_base));
    CHECK(cudaFree(d_step_base));
    CHECK(cudaFree(d_moved_base));
    CHECK(cudaFree(d_max_neighbors_base));
    CHECK(cudaFree(d_nnei_len_base));
    CHECK(cudaFree(d_neighbors_base));
    CHECK(cudaFree(d_prob_array));
    CHECK(cudaFree(d_dist_to_bins));
    CHECK(cudaFree(d_selece_base));
    CHECK(cudaFree(d_probabilities));
    CHECK(cudaFree(d_rec_base));
    CHECK(cudaFree(d_lig_base));

    // 释放 receptor 相关的设备内存
    CHECK(cudaFree(d_receptor_atom_coordinates));
    CHECK(cudaFree(d_ligand_atom_coordinates));
    CHECK(cudaFree(d_current_ligand_reference_pose));
    CHECK(cudaFree(d_receptor_modes));
    CHECK(cudaFree(d_ligand_modes));
    CHECK(cudaFree(d_receptor_mask));
    CHECK(cudaFree(d_ligand_mask));

    // 释放 fastdfire 和 dfire 相关的设备内存
    CHECK(cudaFree(d_fastdfire));
    CHECK(cudaFree(d_dfire_objects_rec));
    CHECK(cudaFree(d_dfire_objects_lig));
    CHECK(cudaFree(d_ener));

}
