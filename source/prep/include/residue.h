#pragma once
#include <cstring>
#include <vector>
#include <iostream>
#include <map>
using namespace std;
struct Residue
{
    string DNA_STANDARD_TYPES[5] = {"DA", "DC", "DI", "DG", "DT"};
    string RNA_STANDARD_TYPES[5] = {"A", "C", "G", "U", "I"};
    string DUMMY_TYPES[2] = {"MMB", "DUM"};
    string name;
    int number;
    string insertion;
    int index;
    int atom_number = 0;
};
