#include <iostream>
#include "swarm_centers.h"



void free_swarmcenters(SwarmCenters &s){
    delete [] s.s;
    delete [] s.pos;
}