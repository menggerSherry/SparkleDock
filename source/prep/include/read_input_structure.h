#pragma once
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include "atom.h"
#include "chain.h"
#include "residue.h"
#include "config_args.h"
#include "complex.h"
using namespace std;
void read_input_structure(
    Complex & complex,
    const char * pdb_file_name,
    bool ignore_oxt,
    bool ignore_hydrogons,
    bool ignore_water,
    bool verbose,
    int N
);


Atom read_atom_line(string line, string line_type, vector<string> residues_to_ignore, vector<string> atoms_to_ignore);

void parse_complex_from_file(
    vector<Atom> & atoms, vector <Chain> & chains, vector <Residue> & residues, string path, vector<string> residues_to_ignore, vector<string> atoms_to_ignore
);


static void trim(string &str){
    string blanks("\f\v\r\t\n ");
    str.erase(0,str.find_first_not_of(blanks));
    str.erase(str.find_last_not_of(blanks) + 1);
}

void get_setup(ConfigArgs & para, string file);

