#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include "atom.h"
#include "chain.h"
#include "residue.h"
using namespace std;

struct Complex{
    vector<Atom> atoms;
    vector<Chain> chain;
    vector<Residue> residue;
    int * atom_index;
    int * residue_index;
    double * atom_coordinates;
    vector<string> elements;
    vector<string> atom_name;
    int coord_size;
    int elements_size;
    int * structure_size; //总的原子数
    double * mass;
    int num_structures;
    int num_atoms;
    int representative_id;
    int protein_num_atoms;
    int nucleic_num_atoms;

    map<string,double> MASSES = {
        {"H", 1.007825},
        {"C", 12.01},
        {"F", 18.9984},
        {"O", 15.9994},
        {"N", 14.0067},
        {"S", 31.972071},
        {"P", 30.973762},
        {"CL", 35.45},
        {"MG", 24.3050},
        {"FE", 55.845},
        {"PB", 207.2},
        {"SE", 78.96}
    } ;   

    // init

    double * translation;

    // residues
    vector<string> residue_name;
    int * residue_id;   
    int * residue_atom_number;  //每个resi种类的原子数
    int * residue_size;
    int num_res_type;


    map<string,string> STANDARD_TYPES = {
        {"ALA", "A"},
        {"ARG", "R"},
        {"ASN", "N"},
        {"ASP", "D"},
        {"CYS", "C"},
        {"GLU", "E"},
        {"GLN", "Q"},
        {"GLY", "G"},
        {"HIS", "H"},
        {"ILE", "I"},
        {"LEU", "L"},
        {"LYS", "K"},
        {"MET", "M"},
        {"PHE", "F"},
        {"PRO", "P"},
        {"SER", "S"},
        {"THR", "T"},
        {"TRP", "W"},
        {"TYR", "Y"},
        {"VAL", "V"}
    };


    map<string, string> MODIFIED_TYPES = {
        {"CYX", "C"}, 
        {"HIP", "H"}, 
        {"HID", "H"}, 
        {"HIE", "H"}
    };
    string DNA_STANDARD_TYPES[5] = {"DA", "DC", "DI", "DG", "DT"};
    string RNA_STANDARD_TYPES[5] = {"A", "C", "G", "U", "I"};
    string surface[26] = {
        "ARG", "ASN", "ASP", "ASX", "CSO", "GLN", 
        "GLU", "GLX", "GLY", "HID", "HIE", "HIP", 
        "IS", "HSD", "HSE", "HSP", "LYS", "PHD", 
        "PRO", "PTR", "SEP", "SER", "THR", "TPO", "TYR", "HIS"};

    // mask 
    int * mask;

    int n_modes;
    // normal modes
    // n_modes * atom_num * 3
    double * modes; 
};


void generate_complex(
    Complex & complex ,vector<vector<Atom>> atom_list, vector<vector<Chain>> chain_list, vector<vector<Residue>> residue_list,
    vector<string> files
);

void free_complex(Complex & complex);


void get_nm_mask(
    int * mask,
    vector<string> residue_names,
    map<string,string> STANDARD_TYPES,
    map<string, string> MODIFIED_TYPES,
    string * DNA_STANDARD_TYPES,
    string * RNA_STANDARD_TYPES,
    int * residue_atom_number,
    int * residue_size
    
);

