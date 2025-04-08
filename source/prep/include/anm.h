#pragma once
#include <cstring>
#include <vector>
#include <iostream>
#include <map>
using namespace std;

struct ANM{
    double * backbone_coord;
    double * hessian;
    double * kirchhoff;
    double * eigvals;
    double * eigvecs;
    double * invvals;
    int n_modes;
    int dof;
    double trace;

};