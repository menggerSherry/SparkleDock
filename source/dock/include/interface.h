#pragma once


struct DockingInterface{
    int * interface;
    int indexes;
};

struct Glowworm
{
    int id = 0;
    double scoring;
    double luciferin = 5.0;
    double rho = 0.4;
    double gamma = 0.6;
    double beta = 0.08;
    double vision_range = 0.2;
    double max_vision_range = 5.0;
    int step = 0;
    int moved = 0;
    int max_neighbors = 5;
    double * reference_points;
    int * neighbors;
    int nnei_len;
    double * current_postion;
};



static unsigned int dist_to_bins[50] = {
         1,  1,  1,  2,  3,  4,  5,  6,  7,  8,
         9, 10, 11, 12, 13, 14, 14, 15, 15, 16,
        16, 17, 17, 18, 18, 19, 19, 20, 20, 21,
        21, 22, 22, 23, 23, 24, 24, 25, 25, 26,
        26, 27, 27, 28, 28, 29, 29, 30, 30, 31};