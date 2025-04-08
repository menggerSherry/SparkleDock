#pragma once
#include <iostream>
#include <vector>
#include "atom.h"
#include "chain.h"
#include "residue.h"
using namespace std;
void translate(
    double * coord, double * translate_center_coo, int * structure_size, 
    const int num_atom, const int num_structure);



