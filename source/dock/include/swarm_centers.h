#pragma once
#include <cstring>
#include <vector>
#include <iostream>
#include <map>
using namespace std;


struct SwarmCenters{
    double *s;  //聚类的元素
    double * pos; //初始化得到的位置 Swarm * glowworm * 10 rec_anm 10 lig_anm 7 平移旋转
    int swarms; //聚类的swarms数量
    int swarm_chunk;
    int swarm_start;
    double  receptor_diameter;
    double ligand_diameter;
    int pos_len;
    int anm_rec;
    int anm_lig;
    
};


void free_swarmcenters(SwarmCenters &s);