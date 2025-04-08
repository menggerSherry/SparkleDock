#pragma once
#include <iostream>
#include <cstring>
using namespace std;
struct ConfigArgs
{
    // argument
    int anm_lig;
    double anm_lig_rmsd;
    int anm_rec;
    double anm_rec_rmsd;
    int anm_seed;
    bool dense_sampling;
    double fixed_distance;
    bool flip;
    int glowworms;
    string ligand_pdb;
    bool membrane;
    bool noh;
    bool now;
    bool noxt;
    string receptor_pdb;
    int starting_points_seed;
    double surface_density;
    double swarm_radius;
    int swarms;
    int swarms_per_restraint;
    bool transmembrane;
    bool use_anm;
    bool verbose_parser;
    bool write_starting_positions;

    double translation_step;
    double rotation_step;
    double nmodes_step;
    bool local_minimization;

    double rho;
    double gamma;
    double vision_range;
    double beta;
    double luciferin;
    int max_neighbor;
    double max_vision_range;
};
