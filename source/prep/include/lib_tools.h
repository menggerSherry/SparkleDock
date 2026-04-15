#pragma once
#include <iostream>
#include <omp.h>
#include <fstream>
static void print_mat(double * mat, int row, int col){
    for(int i = 0; i<row; i++){
        for (int j = 0; j<col; j++){
            std::cout<<mat[i*col+j]<<"  ";
        }
        std::cout<<std::endl;
    }
}



static void print_array(double * mat, int n){
    for(int i = 0; i<n; i++){
        std::cout<<mat[i]<<" ";
    }
    std::cout<<std::endl;
}


static void print_array(int * mat, int n){
    for(int i = 0; i<n; i++){
        std::cout<<mat[i]<<" ";
    }
    std::cout<<std::endl;
}


static bool get_env_cuda(){
    char *var = getenv("USE_CUDA");
    if(var == NULL){
        return false;
    }else{
        return true;
    }
}


static void write_mat(double * mat, int row, int col, string name){
    FILE *fp = NULL;
    fp = fopen(name.c_str(),"w");
    for(int i = 0; i<row; i++){
        for (int j = 0; j<col; j++){
            fprintf(fp, "%f ",mat[i*col+j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

static void write_mati(int * mat, int row, int col, string name){
    FILE *fp = NULL;
    fp = fopen(name.c_str(),"w");
    for(int i = 0; i<row; i++){
        for (int j = 0; j<col; j++){
            fprintf(fp, "%d ",mat[i*col+j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

static void write_matu(unsigned int  * mat, int row, int col, string name){
    FILE *fp = NULL;
    fp = fopen(name.c_str(),"w");
    for(int i = 0; i<row; i++){
        for (int j = 0; j<col; j++){
            fprintf(fp, "%d ",mat[i*col+j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

static void write_matd(int64_t * mat, int row, int col, string name){
    std::ofstream file(name);
    for(int i = 0; i<row; i++){
        for (int j = 0; j<col; j++){
            file <<mat[i*col+j]<<" ";
        }
        file<<std::endl;

    }
    file.close();
}


static int get_env_num_threads(){
    char *var = getenv("NUM_THREADS");
    int ret = 1;
    if(var != NULL){
        ret = atoi(var);
    }
    int numP = omp_get_num_procs();
    // std::cout << "thread num : " << numP << " , " << ret << std::endl;
    return ret;
}


static int get_chunks(){
    char *var = getenv("CHUNKS");
    int ret = 1;
    if(var != NULL){
        ret = atoi(var);
    }
    // int numP = omp_get_num_procs();
    // std::cout << "thread num : " << numP << " , " << ret << std::endl;
    return ret;
}
