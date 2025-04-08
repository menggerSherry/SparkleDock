#pragma once
#include <math.h>

inline double Distance(double * p1, double * p2)
{
	return sqrt((p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]));
}

void kmeans_init_centers(double * center, double * point,int point_number , int K);
void run_kmeans(int * group, double * points,double * centers, int max_iter, int point_number , int K);

inline double Distance3(double * p1, double * p2)
{
	return sqrt((p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]) + (p1[2] - p2[2])*(p1[2] - p2[2]));
}


void kmeans_init_centers3(double * center, double * point,int point_number , int K, int seed);
void run_kmeans3(int * group, double * points,double * centers, int max_iter, int point_number , int K);

