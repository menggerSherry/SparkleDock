#pragma once
#include "complex.h"
#include "anm.h"
#include <iostream>
#include <vector>
#include "swarm_centers.h"

using namespace std;

void cal_anm(Complex & complex, ANM & hessian_type, int n_modes, double rmsd, int seed, int swarms, int glowworm, int rank, int size);

void select_atom_by_name(vector<double> & backbone_coord, vector<int> & backbone_index, double * coord, vector<string> names, string atom_name);

void build_hessian(double ** hessian,double * kirchhoff, double cutoff, double gamma, int n_atoms, double *coords);

void build_hessian1D(double *hessian, double *kirchhoff, double cutoff, double gamma, int n_atoms, double *coords);

void eigh_gpu(double * eigenvalues, double * eigenvectors, double * matrix,int *eigvals, int n_atoms, int rank, int size);

void calModes(ANM  &hessian, int n_modes, bool zeros, bool turbo, int expct_n_zeros, bool reverse, int n_atoms, int rank, int size);

void free_anm(ANM & anm);

void norm(double *array, int size);

void calculate_start_pos(
    SwarmCenters & center,
    Complex & receptor, Complex & ligand, int swarms,int glowworms, int start_points_seed, 
    double * rec_translation,  double * lig_translation,
    double surface_density, bool use_anm, int anm_seed,
    int anm_rec, int anm_lig, bool membrane, bool transmambrane,
    bool write_starting_positions, double swarm_radius, bool flip,
    double fixed_distance, int swarms_per_restraint, bool dense_smapling,const char * name, int rank, int size
);

void calculate_surface_point(
    SwarmCenters & center,
    Complex & receptor, Complex & ligand, int swarms,int glowworms, int start_points_seed, 
    double * rec_translation,  double * lig_translation,
    double surface_density, bool use_anm, int anm_seed,
    int anm_rec, int anm_lig, bool membrane, bool transmambrane,
    bool write_starting_positions, double swarm_radius, bool flip,
    double fixed_distance, int swarms_per_restraint, bool dense_smapling,  const char * rec_name, int number_of_points,int rank, int size
);

int select_surface(double *surface, Complex & complex);


void points_on_sphere(double *sphere_points, int number_of_points);

void cal_dist(double * distances_matrix_rec, double *atom_coordinates, int num_atoms, int dis_size);
void write_center(double * center, std::ofstream &newFile, int swarms);