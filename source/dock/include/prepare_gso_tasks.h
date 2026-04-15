#pragma once
#include "swarm_centers.h"
#include "complex.h"
#include "fastdfire.h"
#include "interface.h"
void get_default_box(double *bounding_box, int anm_rec, int anm_lig, bool use_anm);

void prepare_gso_tasks(
    SwarmCenters & centers, Complex & receptor, Complex &lignad, FastDifire & fastdifire, int seed, double step_translation, double step_rotation, bool use_anm, double nmodes_step,
    int anm_rec, int anm_lig, bool local_minimization, int swarms, int num_glowworms, int rank, int size
);

// void prepare_gso_task_gpu(
//     SwarmCenters & centers,Complex & receptor, Complex &lignad, FastDifire & fastdifire, int seed, double step_translation, double step_rotation, bool use_anm, double nmodes_step,
//     int anm_rec, int anm_lig, bool local_minimization, int swarms, int num_glowworms
// );

void cal_gso_tasks_gpu(SwarmCenters & centers,Complex & receptor, Complex &lignad, FastDifire & fastdifire, int seed, double step_translation, double step_rotation, bool use_anm, double nmodes_step,
    int anm_rec, int anm_lig, bool local_minimization, int swarms, int num_glowworms, double * receptor_reference_points, double * receptor_poles, double * ligand_reference_points, double * ligand_poles,
    int *dfire_objects_rec, int *dfire_objects_lig,
    int object_size_rec, int object_size_lig, int rank, int size);

void cal_gso_tasks_cpu(SwarmCenters & centers,Complex & receptor, Complex &lignad, FastDifire & fastdifire, int seed, double step_translation, double step_rotation, bool use_anm, double nmodes_step,
    int anm_rec, int anm_lig, bool local_minimization, int swarms, int num_glowworms, double * receptor_reference_points, double * receptor_poles, double * ligand_reference_points, double * ligand_poles,
    int *dfire_objects_rec, int *dfire_objects_lig);

void multiplyDiagonalMatrix(int k, int n, double *A, double *B, double * C) ;

void multiplyMatrices(int m, int n, int p, double *A, double *B, double *C);

void transpose(double * mat2, double * mat, int m, int n);

void get_docking_model(int * dfire_objects, Complex & complex, FastDifire & fastdifire);

void free_fastdfire(FastDifire & FastDifire);
void cdfire_calculate_dfire(
    double * energy, int indexes_len,
    int * interface_receptor, int * interface_ligand,  int * rec_objects, 
    int * lig_objects, FastDifire & fastdifire, double * receptor_coordinate, double * ligand_coordinate, 
    int rec_len, int lig_len, int num_glowworms,int swarms);
int select_random_neighbor(
    int * select, double prob, int * neighbors,
    double * probabilities,int select_index,int n, int nnei_len, int swarm
);
void euclidean_dist(
    unsigned int * indexes, int * indexes_len, double * receptor_coordinate, 
    double * ligand_coordinate, int rec_len, int lig_len);

void compute_probability_moving_from_neighbors(
    double * probabilities, int * neighbors, double * luciferin, int nnei_len , double current_luciferin
);
void write_pdb(Complex structure, std::ofstream &newFile, double * rec_copy);
void slerp(double * self, double * other, double rotation_step);
void move_extent(double * self, double * other, int num_ext,double step_nmodes);

void read_nm(double * nm, string file_name);

void rotate(double * coord, double * rotate, int atom_num);

bool compare(Glowworm g1, Glowworm g2);
double cal_rmsd(double * backbone_clusters, double * backbone_glowworm, int cluster_size, int glowworm_size);