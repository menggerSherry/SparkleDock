#pragma once
#include <cstring>
#include <vector>
#include <iostream>
#include <map>
using namespace std;
struct Atom{

    string bacbone_atoms[4] = {"CA", "C", "N", "O"};
    string recgnnized_elements[12] = {"C", "N", "O", "H", "S", "P", "CL", "MG", "FE", "PB", "SE", "F"};
    int number;
    string name;
    string  alternative;
    string chain_id;
    string residue_name;
    int residue_number;
    string residue_insertion;
    float x;
    float y;
    float z;
    float occupancy;
    float b_factor;
    int atom_index;
    string element;
    map<string,float> MASSES = {
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
    bool valid = true;
    string type;

};

struct HetAtom{

    string bacbone_atoms[4] = {"CA", "C", "N", "O"};
    string recgnnized_elements[12] = {"C", "N", "O", "H", "S", "P", "CL", "MG", "FE", "PB", "SE", "F"};
    int number;
    string name;
    string  alternative;
    string chain_id;
    string residue_name;
    int residue_number;
    string residue_insertion;
    float x;
    float y;
    float z;
    float occupancy;
    float b_factor;
    int atom_index;
    map<string,float> MASSES = {
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

};
