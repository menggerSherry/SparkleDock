#include "prepare_gso_tasks.h"
#include "swarm_centers.h"
#include "fastdfire.h"
#include <iostream>
#include <random>
#include <Eigen/Dense>
#include "lib_tools.h"
#include <cmath>
#include <string.h>
#include <fstream>
#include "interface.h"
#include <filesystem>
#include <sstream>
#include <iomanip>  // 包含设置小数点精度的头文件
#include "complex.h"
#include <algorithm>
#include <omp.h>
#include "path_include.h"
#define MAX_TRANSLATION 30
#define MAX_ROTATION 1.0
#define MIN_EXTENT 0.0
#define MAX_EXTENT 1.0
#define DEFAULT_NMODES_REC 10
#define DEFAULT_NMODES_LIG 10

#define STEPS 3
using namespace Eigen;
namespace fs = std::filesystem;
static int num_threads = get_env_num_threads();

void get_docking_model(int * dfire_objects, Complex & complex, FastDifire & fastdifire){
    // 构建dfire 得分的索引，此处检查正确
    int count = 0;
    // #pragma omp parallel for num_threads(num_threads)
    for(int c = 0; c<complex.chain.size();c++){
        for(int r = 0; r<complex.num_res_type;r++){
            int current_atom_index = complex.residue_atom_number[r];
            int num_atom = complex.residue_atom_number[r+1] - complex.residue_atom_number[r];
            
            string current_res_name = complex.residue_name[r];
            // std::cout<<num_atom<<std::endl;
            for (int a = current_atom_index; a < current_atom_index+num_atom; a++)
            {
                string atom_name = complex.atom_name[a];
                string rec_atom_type = current_res_name+ atom_name;
                // std::cout<<rec_atom_type<<std::endl;
                if (rec_atom_type == "MMBBJ"){
                    // TODO: 
                    continue;
                }
                int rnuma, anuma;
                map<string, int>::iterator iter = fastdifire.RES_3.find(current_res_name);
                if(iter != fastdifire.RES_3.end()){
                    rnuma = iter->second;
                }
                map<string, int>::iterator iter_a = fastdifire.atomnumber.find(rec_atom_type);
                if(iter_a != fastdifire.atomnumber.end()){
                    anuma = iter_a->second;
                }
                int atoma = fastdifire.atom_res_trans[rnuma* 14+anuma];
                dfire_objects[count] = atoma;
                count++;

            }
            
        }
    }
    
}

void multiplyDiagonalMatrix(int k, int n, double *A, double *B, double * C) {
    // A diag B: k*n
    // 
    for(int i = 0; i< k;i++){
        // #pragma omp parallel for num_threads(num_threads)
        for(int j = 0; j<n; j++){
            double a_value = A[j];
            C[i*n+j] = B[i*n+j] * a_value;
        }
    } 
    
}

void multiplyMatrices(int m, int n, int p, double *A, double *B, double *C) {
    // 
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[j*n+k];
            }
            C[i*p+j] = sum;
        }
    }
}

void transpose(double * mat2, double * mat, int m, int n){
    // #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<m; i++){
        for(int j = 0; j<n; j++){
            mat2[j*m+i] = mat[i*n+j];
        }
    }
}

void minimum_volume_ellipsoid(
    double * center, double * poles, double * atom_coordinates, double precision, int num_atoms
){
    /*
    计算参考点，此处计算正确， TODO: 但是 docking model没有copy
    */
    // double precision = 0.01;z
    double * Q = new double [num_atoms*4];
    double * QT = new double [num_atoms*4];
    // std::cout<<num_threads<<std::endl;
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<num_atoms; i++){
        double * qt_indices = &QT[i*4];
        double * coords = &atom_coordinates[i*3];
        for(int j = 0;j<4;j++){
            if (j==3)
            {
                qt_indices[j] = 1;
            }else{
                qt_indices[j] = coords[j];
            }
            
        }
    }
    // print_mat(QT,receptor.num_atoms,4);
    // transpose
    // #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<num_atoms; i++){
        for(int j = 0; j<4; j++){
            Q[j*num_atoms+i] = QT[i*4+j];
        }
    }
    // print_mat(Q,4,receptor.num_atoms);
    double * indices = new double [num_atoms*4];
    double * indices2 = new double [num_atoms*4];
    double * M = new double [num_atoms*num_atoms];
    // initialiaze
    double error = 1 + precision;
    double * u = new double [num_atoms];
    int N = num_atoms;
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<N; i++){
        u[i] = 1.0/N;

    }
    // print_array(u,N);
    double *V = new double [4*4];
    double * new_u = new double [N];
    // std::cout<<num_threads<<std::endl;
    while (error>precision)
    {
        // std::cout<<num_threads<<std::endl;
        multiplyDiagonalMatrix(4,N,u,Q,indices);

        // TODO: error here but tolerance
        multiplyMatrices(4,N,4,Q,indices,V);

        MatrixXd v_e = Eigen::Map<Eigen::MatrixXd>(V,4,4);
        MatrixXd Ainv = v_e.inverse();
        double * a_inv = Ainv.data();
        // double* eigMatptrnew = new double[Ainv.size()];
        // Map<MatrixXd>(eigMatptrnew, Ainv.rows(), Ainv.cols()) = Ainv;
        
        multiplyMatrices(4,4,N,a_inv, QT,indices); //4*N
        // print_mat(indices,4,N);
        transpose(indices2,indices,4,N);
        multiplyMatrices(N,4,N,QT, indices2,M);
        double maximum = 0;
        int max_indices = 0;
        for(int i = 0; i<N;i++){
            if(M[i*N+i]>maximum){
                maximum = M[i*N+i];
                max_indices = i;
            }
        }
        // std::cout<<max<<std::endl;
        // std::cout<<max_indices<<std::endl;
        double step_size = (maximum - 3 - 1.0) / ((3 + 1.0) * (maximum - 1.0));
        #pragma omp parallel for num_threads(num_threads)
        for(int i = 0;i<N;i++){
            new_u[i] = (1.0 - step_size) * u[i];
        }
        new_u[max_indices] += step_size;
        double new_error = 0;
        #pragma omp parallel for num_threads(num_threads)
        for(int i = 0;i<N;i++){
            new_error += (new_u[i] - u[i]) * (new_u[i] - u[i]);
        }
        error = sqrt(new_error);
        #pragma omp parallel for num_threads(num_threads)
        for(int i =0;i<N;i++){
            u[i] = new_u[i];
        }

    }
    
    // double * center = new double[3];
    double * pointst = new double [num_atoms*3];
    transpose(pointst,atom_coordinates,num_atoms,3);
    multiplyMatrices(3,N,1,pointst,u,center);
    
    // the A matrix for the ellipsoid
    multiplyDiagonalMatrix(3,N,u,pointst,indices);
    multiplyMatrices(3,N,3,pointst,indices,V);
    for(int i = 0; i<3; i++){
        for(int j = 0; j<3;j++){
            V[i*3+j] -= center[i]*center[j];
        }
    }
    MatrixXd v_e = Eigen::Map<Eigen::MatrixXd>(V,3,3);
    MatrixXd Ainv = v_e.inverse();
    double * A = Ainv.data();
    for (int i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            A[i*3+j] /= 3;
        }
        
    }

    MatrixXd A_e = Eigen::Map<Eigen::MatrixXd>(A,3,3);

    JacobiSVD<Eigen::MatrixXd> svd(A_e, ComputeFullU | ComputeFullV );  
    MatrixXd rotation = svd.matrixV(), s = svd.singularValues();
    double * rotation_array = rotation.data();
    double * radii = s.data();
    for(int i = 0; i<3; i++){
        radii[i] = 1.0/sqrt(radii[i]);
    }
    double * axes = new double [3*3];
    multiplyDiagonalMatrix(3,3,radii,rotation_array,axes);
    // double * poles = new double [3*6];
    int index = 0;
    for(int i =0; i<3;i++){
        double x1 = -axes[i*3] + center[0];
        double x2 = axes[i*3] + center[0];
        double y1 = -axes[i*3+1] + center[1];
        double y2 = axes[i*3+1] + center[1];
        double z1 = -axes[i*3+2] + center[2];
        double z2 = axes[i*3+2] + center[2];
        poles[index*3] =x1;
        poles[index*3+1] = y1;
        poles[index*3+2] = z1;
        index++;
        poles[index*3] =x2;
        poles[index*3+1] = y2;
        poles[index*3+2] = z2;
        
    }
    delete [] Q;
    delete [] QT;
    delete [] indices;
    delete [] indices2;
    delete [] M;
    delete [] u;
    delete [] new_u;
    delete [] V;
    delete [] pointst;
    delete [] axes;

}

void qmul(double *Q, double * self, double * other){
    Q[0] = (self[0] * other[0] - self[1] * other[1] - self[2] * other[2] - self[3] * other[3]);
    Q[1] = (self[0] * other[1] + self[1] * other[0] + self[2] * other[3] - self[3] * other[2]);
    Q[2] = (self[0] * other[2] - self[1] * other[3] + self[2] * other[0] + self[3] * other[1]);
    Q[3] = (self[0] * other[3] + self[1] * other[2] - self[2] * other[1] + self[3] * other[0]);
}


void rotate(double * coord, double * rotate, int atom_num)
{
    // rotation
                
    double weight = rotate[0];
    double pos_x = rotate[1];
    double pos_y = rotate[2];
    double pos_z = rotate[3];

    

    double rotate_norm2 = weight * weight + pos_x * pos_x + pos_y * pos_y + pos_z * pos_z;
    // std::cout<<"normsss"<<rotate_norm2<<std::endl;
    // conjunct
    double * rotate_inverse = new double [4];
    double * current_pos = new double[4];
    double * current_pos2 = new double [4];
    // current_pos[0] = 0;
    rotate_inverse[0] = rotate[0] / rotate_norm2;
    rotate_inverse[1] = - rotate[1] / rotate_norm2;
    rotate_inverse[2] = - rotate[2] / rotate_norm2;
    rotate_inverse[3] = - rotate[3] / rotate_norm2;
    

    for(int i = 0; i<atom_num; i++){
        current_pos[0] = 0;
        // double w = 0 ;
        current_pos[1] = coord[i*3];
        current_pos[2] = coord[i*3+1];
        current_pos[3] = coord[i*3+2];
        // memcpy(current_pos+1, &coord[i*3], sizeof(double)*3);

        // qmul(current_pos2,rotate,current_pos);
        

        // qmul(current_pos, current_pos2, rotate_inverse);
        // qmul(current_pos2,rotate,current_pos);
        current_pos2[0] = (rotate[0] * current_pos[0] - rotate[1] * current_pos[1] - rotate[2] * current_pos[2] - rotate[3] * current_pos[3]);
        current_pos2[1] = (rotate[0] * current_pos[1] + rotate[1] * current_pos[0] + rotate[2] * current_pos[3] - rotate[3] * current_pos[2]);
        current_pos2[2] = (rotate[0] * current_pos[2] - rotate[1] * current_pos[3] + rotate[2] * current_pos[0] + rotate[3] * current_pos[1]);
        current_pos2[3] = (rotate[0] * current_pos[3] + rotate[1] * current_pos[2] - rotate[2] * current_pos[1] + rotate[3] * current_pos[0]);
        

        // qmul(current_pos, current_pos2, rotate_inverse);
        current_pos[0] = (current_pos2[0] * rotate_inverse[0] - current_pos2[1] * rotate_inverse[1] - current_pos2[2] * rotate_inverse[2] - current_pos2[3] * rotate_inverse[3]);
        current_pos[1] = (current_pos2[0] * rotate_inverse[1] + current_pos2[1] * rotate_inverse[0] + current_pos2[2] * rotate_inverse[3] - current_pos2[3] * rotate_inverse[2]);
        current_pos[2] = (current_pos2[0] * rotate_inverse[2] - current_pos2[1] * rotate_inverse[3] + current_pos2[2] * rotate_inverse[0] + current_pos2[3] * rotate_inverse[1]);
        current_pos[3] = (current_pos2[0] * rotate_inverse[3] + current_pos2[1] * rotate_inverse[2] - current_pos2[2] * rotate_inverse[1] + current_pos2[3] * rotate_inverse[0]);
        
        // std::cout<<current_pos[0]<<" "<<current_pos[1]<<" "<<current_pos[2]<<" "<<current_pos[3]<<std::endl;
        coord[i*3] = current_pos[1];
        coord[i*3+1] = current_pos[2];
        coord[i*3+2] = current_pos[3];

        // memcpy(&coord[i*3], &current_pos[1], sizeof(double)*3);
        
    }
    delete [] rotate_inverse;
    delete [] current_pos;
    delete [] current_pos2;

}

void read_nm(double * nm, string file_name){
    std::ifstream ifs(file_name.c_str());
    if (!ifs.is_open()) {
        std::cerr << "Error opening file: DCparams"  << std::endl;
        exit(0);
    }
    std::string line;
    int count = 0;
    while (std::getline(ifs, line)) {
        double value = std::stod(line);
        nm[count] = value;
        
        // std::cout<<fastdifire.difire_energy[count]<<std::endl;
        count++;
    }
}



void cal_gso_tasks_cpu(SwarmCenters & centers,Complex & receptor, Complex &lignad, FastDifire & fastdifire, int seed, double step_translation, double step_rotation, bool use_anm, double nmodes_step,
    int anm_rec, int anm_lig, bool local_minimization, int swarms, int num_glowworms, double * receptor_reference_points, double * receptor_poles, double * ligand_reference_points, double * ligand_poles,
    int *dfire_objects_rec, int *dfire_objects_lig)
{
    int num_threads = get_env_num_threads();
    DockingInterface rec_interface, lig_interface;
    
    // // run dock

    // Glowworm * glowworms = new Glowworm[swarms*num_glowworms];
    // glowworm  soa
    int * id_base = new int [swarms*num_glowworms];
    double * scoring_base = new double[swarms*num_glowworms];
    double * luciferin_base = new double [swarms*num_glowworms];
    double * rho_base = new double [swarms*num_glowworms];
    double * gamma_base = new double [swarms*num_glowworms];
    double * beta_base = new double [swarms*num_glowworms];
    double * vision_range_base = new double[swarms*num_glowworms];
    double * max_vision_range_base = new double [swarms*num_glowworms];
    int * step_base = new int  [swarms*num_glowworms];
    int * moved_base  = new int  [swarms*num_glowworms];
    int * max_neighbors_base = new int [swarms*num_glowworms];
    int * nnei_len_base = new int[swarms*num_glowworms];
    int * neighbors_base = new int [swarms*num_glowworms*num_glowworms];

    // 赋值

    for (int i = 0; i < swarms*num_glowworms; i++)
    {
        id_base[i] = 0;
        scoring_base[i] = 0;
        luciferin_base[i] = 5.0;
        rho_base[i] = 0.4;
        gamma_base[i] = 0.6;
        beta_base[i] = 0.08;
        vision_range_base[i] = 0.2;
        max_vision_range_base[i] = 5.0;
        step_base[i] = 0;
        moved_base[i] = 0;
        max_neighbors_base[i] = 5;
        nnei_len_base[i] = 0;
    }
    clock_t start, finish;
    double duration = 0;
    double * receptor_pose = new double [swarms*receptor.num_atoms*3];
    double * ligand_pose = new double [swarms*lignad.num_atoms*3];
    double * ligand_reference_pose = new double[swarms*num_glowworms*3];
    double * rec_copy_base = new double [swarms*num_glowworms*receptor.num_atoms*3];
    double * lig_copy_base = new double [swarms*num_glowworms*lignad.num_atoms*3];
    int * select_base = new int [swarms * num_glowworms];
    double * select_postition_base = new double [swarms * num_glowworms*(7+centers.anm_rec+centers.anm_lig)];
    double * delta_base = new double [swarms * 3];
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::mt19937 gen(0);
    for(int i = 0; i<swarms; i++){
        for(int g = 0; g<num_glowworms; g++){
            id_base[i*num_glowworms+g] = g;
        }
    }
    // 构造 random array
    double * prob_array = new double [STEPS * swarms * num_glowworms];
    for (int i = 0; i< swarms; i++){
        for(int s = 0; s<STEPS; s++){
            for(int j = 0; j<num_glowworms; j++ ){
                double value  = distribution(gen);
                prob_array[i*STEPS*num_glowworms + s*num_glowworms + j] = value;
            }
        }
    }
    // std::cout<<PROJECT_PATH<<std::endl;
    for(int step = 0; step<STEPS; step++){
        std::cout<<"step:"<<step<<std::endl;
        start = clock();
        #pragma omp parallel for num_threads(num_threads)
        for(int i = 0; i<swarms; i++){
            double *current_pos = &centers.pos[i*centers.pos_len];
            // 
            double *current_receptor_pose = &receptor_pose[i*receptor.num_atoms*3];
            double *current_ligand_pose = &ligand_pose[i*lignad.num_atoms*3];
            double * rec_copy = &rec_copy_base[i*num_glowworms*receptor.num_atoms*3];
            double * lig_copy = &lig_copy_base[i*num_glowworms*lignad.num_atoms*3];
            // double * rec_copy = new double [num_glowworms*receptor.num_atoms*3];
            // double * lig_copy = new double [num_glowworms*lignad.num_atoms*3];
            
            int * select = &select_base[i*num_glowworms];
            // int * select = new int[num_glowworms];
            int select_index = 0;
            double * select_postion = &select_postition_base[i*num_glowworms*(7+centers.anm_rec+centers.anm_lig)];
            double * delta = &delta_base[i*3];

            for(int g = 0 ; g<num_glowworms; g++){
                // int id = id_base[i*num_glowworms+g];
                
                // Glowworm *current_glowworm = &glowworms[i*num_glowworms+g];
                double * pos_gloworm = &current_pos[g*(7+centers.anm_rec+centers.anm_lig)];
                double * current_ligand_reference_pose = &ligand_reference_pose[i*num_glowworms*3+g*3];


                if (moved_base[i*num_glowworms+g] || step_base[i*num_glowworms+g]==0){
                    memcpy(current_receptor_pose, receptor.atom_coordinates, sizeof(double)*receptor.num_atoms*3);
                    memcpy(current_ligand_pose, lignad.atom_coordinates, sizeof(double)*lignad.num_atoms*3);
                    memcpy(current_ligand_reference_pose, ligand_reference_points, sizeof(double)*3);
                    
                    


                    if(DEFAULT_NMODES_REC>0){

                        for(int j = 0;j<DEFAULT_NMODES_REC; j++){
                            // DEFAULT_NMODES_REC * num_atoms * 3 TODO: 查看一下nmode
                            double * rep_modes = &receptor.modes[j*receptor.num_atoms*3];
                            double rec_instant = pos_gloworm[7+j];
                            for(int p = 0; p<receptor.num_atoms; p++){

                                int mask  = receptor.mask[p];
                                if(mask == 1){
                                    current_receptor_pose[p*3] += rep_modes[p*3] * rec_instant;
                                    current_receptor_pose[p*3+1] += rep_modes[p*3+1] * rec_instant;
                                    current_receptor_pose[p*3+2] += rep_modes[p*3+2] * rec_instant;
                                }
                            }
    
                        }
                    }
                    if(i==3 && g==3){
                        printf("%f %f %f \n", current_receptor_pose[0],current_receptor_pose[1],current_receptor_pose[2]);
                    }
                    
                    if(DEFAULT_NMODES_LIG >0){

                        for(int j = 0;j<DEFAULT_NMODES_LIG; j++){

                            double * rep_modes = &lignad.modes[j*lignad.num_atoms*3];
                            double rec_instant = pos_gloworm[7+DEFAULT_NMODES_REC+j];
                            for(int p = 0; p<lignad.num_atoms; p++){

                                int mask  = lignad.mask[p];
                                if(mask == 1){
                                    current_ligand_pose[p*3] += rep_modes[p*3] * rec_instant;
                                    current_ligand_pose[p*3+1] += rep_modes[p*3+1] * rec_instant;
                                    current_ligand_pose[p*3+2] += rep_modes[p*3+2] * rec_instant;
                                }
                            }
    
                        }
                    }
                    
                    // if(i==0 && g==1){
                    //     printf("%f %f %f \n", current_ligand_pose[0],current_ligand_pose[1],current_ligand_pose[2]);
                    //     printf("%f %f %f \n", current_receptor_pose[0],current_receptor_pose[1],current_receptor_pose[2]);
        
                    // }

                    rotate(current_ligand_pose,&pos_gloworm[3],lignad.num_atoms);
                    rotate(current_ligand_reference_pose, &pos_gloworm[3], 1);
                    

                    

                    for(int ii = 0; ii<lignad.num_atoms; ii++){
                        current_ligand_pose[ii*3] += pos_gloworm[0];
                        current_ligand_pose[ii*3+1] +=  pos_gloworm[1];
                        current_ligand_pose[ii*3+2] +=  pos_gloworm[2];
                    }

                    current_ligand_reference_pose[0] += pos_gloworm[0];
                    current_ligand_reference_pose[1] += pos_gloworm[1];
                    current_ligand_reference_pose[2] += pos_gloworm[2];
                    if(i==3 && g==3){
                        printf("%f %f %f \n", current_ligand_pose[0],current_ligand_pose[1],current_ligand_pose[2]);
                        printf("%f %f %f \n", current_ligand_reference_pose[0],current_ligand_reference_pose[1],current_ligand_reference_pose[2]);
                    }
                    // TODO: free
                    // current_glowworm->reference_points = current_ligand_reference_pose;
                    
                    
                    
                    int index_len;
                    double energy = 0.0;
                    // call objective function
                    // print_mat(receptor_pose,receptor.num_atoms,3);
                    
                    cdfire_calculate_dfire(
                    &energy, index_len,
                    rec_interface.interface,lig_interface.interface,
                    dfire_objects_rec,dfire_objects_lig,fastdifire,current_receptor_pose,
                    current_ligand_pose,receptor.num_atoms,lignad.num_atoms,g,i);
                    // std::cout<<index_len<<std::endl;
                    if(i==3 && g==3){
                        printf("%f \n", energy);
                    }
                    scoring_base[i*num_glowworms+g] = energy;
                    // std::cout<<scoring_base[i*num_glowworms+g]<<std::endl;
                //     if(i==0 && g==62){
                //     std::cout<<"l:"<<scoring_base[i*num_glowworms+g]<<std::endl;
                
                // }
                }
                // std::cout<<luciferin_base[i*num_glowworms+g]<<std::endl;
                luciferin_base[i*num_glowworms+g] = (1.0 - rho_base[i*num_glowworms+g]) * luciferin_base[i*num_glowworms+g] + gamma_base[i*num_glowworms+g] * scoring_base[i*num_glowworms+g];
                step_base[i*num_glowworms+g] +=1;

                // if(i==0 && g==62){
                //     printf("luciferin reference:%f %f %f \n",current_ligand_reference_pose[0],current_ligand_reference_pose[1],current_ligand_reference_pose[2]);
    
                //     std::cout<<"l:"<<luciferin_base[i*num_glowworms+g]<<std::endl;
                
                // }
                // exit(-1);
            }

        }

        finish = clock();
        duration += (double)(finish - start) / CLOCKS_PER_SEC;
        
        // if(step==90){
        //     write_mat(luciferin_base, swarms, num_glowworms, "cpu_luciferin.txt");
        // }
        // 
        // write_mat(luciferin_base, swarms, num_glowworms, "cpu_luciferin.txt");
        // exit(-1);

        // receptor pose, ligand pose,  luciferin, 

        #pragma omp parallel for num_threads(num_threads)
        for(int i = 0; i<swarms; i++){
            double *current_pos = &centers.pos[i*centers.pos_len];
            double *current_receptor_pose = &receptor_pose[i*receptor.num_atoms*3];
            double *current_ligand_pose = &ligand_pose[i*lignad.num_atoms*3];
            double * rec_copy = &rec_copy_base[i*num_glowworms*receptor.num_atoms*3];
            double * lig_copy = &lig_copy_base[i*num_glowworms*lignad.num_atoms*3];
            // double * rec_copy = new double [num_glowworms*receptor.num_atoms*3];
            // double * lig_copy = new double [num_glowworms*lignad.num_atoms*3];
            
            int * select = &select_base[i*num_glowworms];
            // int * select = new int[num_glowworms];
            int select_index = 0;
            double * select_postion = &select_postition_base[i*num_glowworms*(7+centers.anm_rec+centers.anm_lig)];
            double * delta = &delta_base[i*3];
            // std::cout<<"pass"<<std::endl;
            for(int g = 0; g<num_glowworms; g++){
                // Glowworm *current_glowworm = &glowworms[i*num_glowworms+g];
                double * current_reference = &ligand_reference_pose[i*num_glowworms*3+g*3];

                int * neighbors = &neighbors_base[i*num_glowworms*num_glowworms+g*num_glowworms];
                memset(neighbors,0,sizeof(int)*num_glowworms);
                double squared_vision_range = vision_range_base[i*num_glowworms+g] * vision_range_base[i*num_glowworms+g];
                // neighbor list
                nnei_len_base[i*num_glowworms+g] = 0; 
                // int cc =0;
                if(i==0 && g==95){
                    printf("improtant nl:%.10f\n",luciferin_base[i*num_glowworms + g]);
                    printf("%f %f %f \n",current_reference[0],current_reference[1],current_reference[2]);
                    printf("%f \n", vision_range_base[i*num_glowworms+g]);
                }
                for(int n = 0; n<num_glowworms; n++){
                    // Glowworm *neighbor = &glowworms[i*num_glowworms+n];
                    
                    // if(i==0 && g==56 && n==19){
                    //         printf("%d \n",n);
                    //         printf("%f \n",luciferin_base[i*num_glowworms+n]);
                    // }
                    if(n!=g && luciferin_base[i*num_glowworms+g] < luciferin_base[i*num_glowworms+n]){
                        // cal distance 
                        double * neighbor_reference =  &ligand_reference_pose[i*num_glowworms*3+n*3];
                        // if(i==0 && g==56){
                        //     printf("%d \n",n);
                        // }
                        // exit(-1);
                        double distance = 0;
                        for(int dis = 0; dis < 3; dis++){
                            distance += ((current_reference[dis] - neighbor_reference[dis])*(current_reference[dis] - neighbor_reference[dis]));
                        }
                        // std::cout<<"dis"<<distance<<std::endl;
                        if (distance < squared_vision_range){
                            // neighbors 
                            neighbors[nnei_len_base[i*num_glowworms+g]++] = n;
                        }

                    }
                }
                if(i==0 && g==95){
                    printf("nnei len %d\n",nnei_len_base[i*num_glowworms+g]);
                    for(int ii = 0; ii< nnei_len_base[i*num_glowworms+g]; ii++){
                        printf("%d ",neighbors[ii]);
                    }
                    printf("\n");
                }
               
                double total_sum = 0;
                double * probabilities = new double [nnei_len_base[i*num_glowworms+g]];
                // std::cout<<11111<<std::endl;
                compute_probability_moving_from_neighbors(probabilities, neighbors, &luciferin_base[i*num_glowworms], nnei_len_base[i*num_glowworms+g],  luciferin_base[i*num_glowworms+g]);

                // double prob = random_seed[g];
                double prob = prob_array[i*STEPS*num_glowworms + step*num_glowworms + g];
                // g: current glowworm
                // select_index = select_random_neighbor(select, prob, current_glowworm,probabilities,select_index,g);
                
                // int nnei_len = nnei_len_base[i*num_glowworms+g];
                // if(nnei_len==0){
                //     select[g] = g;
                // }
                // else{
                //     double sum_probabilities = 0.0; 
                //     int ii;
                //     for(ii = 0; ii<nnei_len; ii++){
                //         sum_probabilities += probabilities[ii];
                //         if(sum_probabilities>=prob) break;
                //     }

                //     select[g] = neighbors[ii-1];
                    
    

                // }
                select_index = select_random_neighbor(select, prob, neighbors, probabilities,select_index,g, nnei_len_base[i*num_glowworms+g],i);
                // 获得当前的nnei_index
                int nnei_index = select[g];
                if(i==0 && g==95){
                    printf("nnei_index %d\n",nnei_index);
                    printf("prob %f \n",prob);
                    // printf("move_postion:%f %f %f \n",current_pos[nnei_index*(7+centers.anm_rec+centers.anm_lig)],current_pos[nnei_index*(7+centers.anm_rec+centers.anm_lig)+1],current_pos[nnei_index*(7+centers.anm_rec+centers.anm_lig)+2]);
        
                }

                double * move_postion = &select_postion[g*(7+centers.anm_rec+centers.anm_lig)]; //
                
                
                memcpy(
                    move_postion, 
                    &current_pos[nnei_index*(7+centers.anm_rec+centers.anm_lig)],
                    sizeof(double)*(7+centers.anm_rec+centers.anm_lig)
                    );
                // copy

                // if(i==0 && g==56){
                //     printf("move_postion111:%f %f %f \n",move_postion[0],move_postion[1],move_postion[2]);
                //     printf("move_translation2:%f %f %f \n",current_pos[nnei_index*(7+centers.anm_rec+centers.anm_lig) + 0],current_pos[nnei_index*(7+centers.anm_rec+centers.anm_lig) + 1],current_pos[nnei_index*(7+centers.anm_rec+centers.anm_lig) + 2]);
    
                //     // printf("move_translation2:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
                
                // }
                // delete [] current_glowworm->neighbors;
                delete [] probabilities;


            }
        }
        
        // if(step==6){
        //     exit(0);
        // }
        // write_mat(select_postition_base, swarms, num_glowworms*(7+centers.anm_rec+centers.anm_lig), "gpu_luciferin.txt");
        // exit(-1);

        #pragma omp parallel for num_threads(num_threads)
        for(int i = 0; i<swarms; i++){
            double *current_pos = &centers.pos[i*centers.pos_len];
            double *current_receptor_pose = &receptor_pose[i*receptor.num_atoms*3];
            double *current_ligand_pose = &ligand_pose[i*lignad.num_atoms*3];
            double * rec_copy = &rec_copy_base[i*num_glowworms*receptor.num_atoms*3];
            double * lig_copy = &lig_copy_base[i*num_glowworms*lignad.num_atoms*3];
            // double * rec_copy = new double [num_glowworms*receptor.num_atoms*3];
            // double * lig_copy = new double [num_glowworms*lignad.num_atoms*3];
            
            int * select = &select_base[i*num_glowworms];
            // int * select = new int[num_glowworms];
            int select_index = 0;
            double * select_postion = &select_postition_base[i*num_glowworms*(7+centers.anm_rec+centers.anm_lig)];
            double * delta = &delta_base[i*3];
            for(int g = 0; g < num_glowworms; g++){

                // Glowworm *current_glowworm = &glowworms[i*num_glowworms+g];
                int nei_idx = select[g]; //选择对应的neighbor
                // Glowworm *neighbor_glowworm = &glowworms[i*num_glowworms+nei_idx];
                // std::cout<<nei_idx<<" "<<g<<std::endl;

                double * current_postion = &current_pos[g*(7+centers.anm_rec+centers.anm_lig)];
                double * move_postion = &select_postion[g*(7+centers.anm_rec+centers.anm_lig)];

                moved_base[i*num_glowworms+g] = id_base[i*num_glowworms+g] != id_base[i*num_glowworms+nei_idx];
                // if(i==0 && g==62){
                //     printf("%d\n",id_base[i*num_glowworms+g]);
                //     printf("%d\n",id_base[i*num_glowworms+nei_idx]);
                // }
                if(id_base[i*num_glowworms+g] != id_base[i*num_glowworms+nei_idx]){

                    double * move_tranlation = &move_postion[0];
                    double * current_translation = &current_postion[0];
                    double * current_rotation = &current_postion[3];
                    double * move_rotation = &move_postion[3];

                    double * current_rec_extent = &current_postion[7];
                    double * move_rec_extent = &move_postion[7];

                    double * current_lig_extent = &current_postion[7+centers.anm_rec];
                    double * move_lig_extent = &move_postion[7+centers.anm_rec];


                    double n = 0;
                    // if(i==0 && g==62){
                    //     printf("move_translation:%f %f %f \n",move_tranlation[0],move_tranlation[1],move_tranlation[2]);
                    //     printf("move_translation2:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
        
                    // }
                    // TODO: replace i
                    if(i==0 && g==95){
                        printf("current_translation:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
                        printf("move:%f %f %f \n",move_tranlation[0],move_tranlation[1],move_tranlation[2]);
        
                    }
                    for(int d = 0;d<3;d++){
                        delta[d] = move_tranlation[d] - current_translation[d];
                        n+=(delta[d] * delta[d]);
                        // std::cout<<delta[d]<<std::endl;
                    }
                    n = sqrt(n);
                    // if(i==0 && g==62){
                    //     printf("n: %f", n);
                    // }
                    if (n>=1e-8){
                        for(int d = 0; d<3; d++){
                            delta[d] *= (0.5 / n);
                            current_translation[d] += delta[d];
                        }
                    }      
                    
                        // printf("current_translation:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
                        
                    
                    slerp(current_rotation, move_rotation, 0.5);

                    move_extent(current_rec_extent,move_rec_extent,centers.anm_rec,0.5);
                    move_extent(current_lig_extent,move_lig_extent,centers.anm_lig,0.5);
                    // if(i==0 && g==56){
                    // printf("current_translation:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
                    // printf("current_rotation:%f %f %f %f \n",current_rotation[0],current_rotation[1],current_rotation[2], current_rotation[3]);
                    // printf("rec_ext:%f %f %f %f \n",current_rec_extent[0],current_rec_extent[1],current_rec_extent[2], current_rec_extent[3]);
                    // printf("lig_ext:%f %f %f %f \n",current_lig_extent[0],current_lig_extent[1],current_lig_extent[2], current_lig_extent[3]);
                    // // printf("current_translation:%f %f %f \n",current_translation[0],current_translation[1],current_translation[2]);
                    // }

                }
                // update conformers
                // update vision range
                vision_range_base[i*num_glowworms+g] = vision_range_base[i*num_glowworms+g] + beta_base[i*num_glowworms+g] * (max_neighbors_base[i*num_glowworms+g] - nnei_len_base[i*num_glowworms+g]);
                if(vision_range_base[i*num_glowworms+g]<0.0){vision_range_base[i*num_glowworms+g] = 0;}
                if(vision_range_base[i*num_glowworms+g] > max_vision_range_base[i*num_glowworms+g]){vision_range_base[i*num_glowworms+g]=max_vision_range_base[i*num_glowworms+g];}
                // if(i==0 && g==56){
                //     printf("vision range%f\n",vision_range_base[i*num_glowworms+g]);
                //     // printf("%d\n",nei_id);
                // }
            }
        }
        // if(step==10){
        // exit(0);
        // }
    
    }
    printf( "%f seconds\n", duration );
    // write_mat(centers.pos, swarms, centers.pos_len, "cpu_luciferin.txt");
    exit(0);

    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<swarms; i++){


        double *current_pos = &centers.pos[i*centers.pos_len];
        double *current_receptor_pose = &receptor_pose[i*receptor.num_atoms*3];
        double *current_ligand_pose = &ligand_pose[i*lignad.num_atoms*3];
        double * rec_copy = &rec_copy_base[i*num_glowworms*receptor.num_atoms*3];
        double * lig_copy = &lig_copy_base[i*num_glowworms*lignad.num_atoms*3];
        // double * rec_copy = new double [num_glowworms*receptor.num_atoms*3];
        // double * lig_copy = new double [num_glowworms*lignad.num_atoms*3];
        
        int * select = &select_base[i*num_glowworms];
        // int * select = new int[num_glowworms];
        int select_index = 0;
        double * select_postion = &select_postition_base[i*num_glowworms*(7+centers.anm_rec+centers.anm_lig)];
        double * delta = &delta_base[i*3];


        string save_swarm = "swarm_";
        string file_name = "gso_";
        string swarms = to_string(i);
        string steps = to_string(STEPS) ;
        string file_type = ".out";
        string save_dir =  save_swarm+swarms;
        string save_file = file_name+steps+file_type;
        // std::cout<<save_file<<std::endl;
        string slash = "/";
        string full_path = save_dir+slash+save_file;
        std::ofstream newFile(full_path.c_str());

        if (newFile.is_open()) {
            // 写入文件内容
            newFile << "#Coordinates  RecID  LigID  Luciferin  Neighbor's number  Vision Range  Scoring\n";
            // 可以继续在此处写入其他内容
            // omp
            
            for(int g = 0; g<num_glowworms;g++){
                newFile<<"(";
                // Glowworm *current_glowworm =&glowworms[i*num_glowworms+g];
                double * current_postion = &current_pos[g*(7+centers.anm_rec+centers.anm_lig)];

                for(int pp = 0; pp < (7+centers.anm_rec+centers.anm_lig); pp++){
                    newFile << current_postion[pp];
                    if(pp<(7+centers.anm_rec+centers.anm_lig)-1){
                        newFile << ", ";
                    }
                    
                }
                newFile<<")\t";
                newFile<<0<<"\t";
                newFile<<0<<"\t";
                newFile<<luciferin_base[i*num_glowworms+g]<<"\t";
                newFile<<nnei_len_base[i*num_glowworms+g]<<"\t";
                newFile<<vision_range_base[i*num_glowworms+g]<<"\t";
                newFile<<scoring_base[i*num_glowworms+g]<<" ";
                newFile<<"\n";
            }
            

            // 关闭文件
            newFile.close();
        } else {
            std::cerr << "Unable to create the file." << std::endl;
        }
        

        for(int g = 0; g<num_glowworms; g++){
            double * current_rec_copy = &rec_copy[g*receptor.num_atoms*3];
            double * current_lig_copy = &lig_copy[g*lignad.num_atoms*3];

            memcpy(&current_rec_copy[0],&receptor.atom_coordinates[0],sizeof(double)*receptor.num_atoms*3);
            memcpy(&current_lig_copy[0],&lignad.atom_coordinates[0],sizeof(double)*lignad.num_atoms*3);

            double * current_postion = &current_pos[g*(7+centers.anm_rec+centers.anm_lig)];

            // print_mat(current_rec_copy,receptor.num_atoms,3);
            // print_mat(current_lig_copy,lignad.num_atoms,3);
            // exit(-1);
            // use normal modes
            for(int m = 0; m<receptor.n_modes; m++){
                double * current_modes_rec = &receptor.modes[m*receptor.num_atoms*3];
                double rec_extent = current_postion[7+m];
                
                for(int dd = 0; dd<receptor.num_atoms; dd++){
                    if(receptor.mask[dd]==1){
                        current_rec_copy[dd*3] += current_modes_rec[dd*3] * rec_extent;
                        current_rec_copy[dd*3+1] += current_modes_rec[dd*3+1] * rec_extent;
                        current_rec_copy[dd*3+2] += current_modes_rec[dd*3+2] * rec_extent;
                    }
                }

            }

            for(int m = 0; m<lignad.n_modes; m++){
                double * current_modes_lig = &lignad.modes[m*lignad.num_atoms*3];
                double lig_extent = current_postion[7+centers.anm_rec+m];
                
                for(int dd = 0; dd<lignad.num_atoms; dd++){
                    if(lignad.mask[dd]==1){
                        current_lig_copy[dd*3] += current_modes_lig[dd*3] * lig_extent;
                        current_lig_copy[dd*3+1] += current_modes_lig[dd*3+1] * lig_extent;
                        current_lig_copy[dd*3+2] += current_modes_lig[dd*3+2] * lig_extent;
                    }
                }

            }
            // rotate and translate
            rotate(current_lig_copy,&current_postion[3],lignad.num_atoms);
            for(int ii = 0; ii<lignad.num_atoms; ii++){
                current_lig_copy[ii*3] += current_postion[0];
                current_lig_copy[ii*3+1] +=  current_postion[1];
                current_lig_copy[ii*3+2] +=  current_postion[2];
            }
        }

        for(int g = 0; g<num_glowworms; g++){
            double * current_rec_copy = &rec_copy[g*receptor.num_atoms*3];
            double * current_lig_copy = &lig_copy[g*lignad.num_atoms*3];
            string save_swarm = "swarm_";
            string file_name = "lightdock_";
            string swarms = to_string(i);
            string steps = to_string(g) ;
            string file_type = ".pdb";
            string save_dir =  save_swarm+swarms;
            string save_file = file_name+steps+file_type;
            // std::cout<<save_file<<std::endl;
            string slash = "/";
            string full_path = save_dir+slash+save_file;
            // TODO: 写文件
            std::ofstream newFile(full_path.c_str());
            write_pdb(receptor,newFile,current_rec_copy);
            write_pdb(lignad,newFile,current_lig_copy);
            newFile.close();
        }
    }

    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<swarms; i++){
        double *current_pos = &centers.pos[i*centers.pos_len];
        double *current_receptor_pose = &receptor_pose[i*receptor.num_atoms*3];
        double *current_ligand_pose = &ligand_pose[i*lignad.num_atoms*3];
        double * rec_copy = &rec_copy_base[i*num_glowworms*receptor.num_atoms*3];
        double * lig_copy = &lig_copy_base[i*num_glowworms*lignad.num_atoms*3];
        // double * rec_copy = new double [num_glowworms*receptor.num_atoms*3];
        // double * lig_copy = new double [num_glowworms*lignad.num_atoms*3];
        
        int * select = &select_base[i*num_glowworms];
        // int * select = new int[num_glowworms];
        int select_index = 0;
        double * select_postion = &select_postition_base[i*num_glowworms*(7+centers.anm_rec+centers.anm_lig)];
        double * delta = &delta_base[i*3];

        // 聚类
        Glowworm * sorted_glowworms = new Glowworm [num_glowworms];
        for(int g = 0; g<num_glowworms; g++){
            sorted_glowworms[g].id = id_base[i*num_glowworms+g];
        }
        for(int g = 0; g<num_glowworms; g++){
            sorted_glowworms[g].scoring = scoring_base[i*num_glowworms+g];
        }
        // memcpy(sorted_glowworms, glowworms+i*num_glowworms, sizeof(Glowworm)*num_glowworms);
        // 按照glowworm的id排序
        std::sort(sorted_glowworms,sorted_glowworms+num_glowworms,compare);
        // for(int i = 0; i<num_glowworms; i++){
        //     std::cout<<sorted_glowworms[i].id<<std::endl;
        // }
        // print()
        // 列类别， 行属于该类别的索引
        int * cluster = new int [num_glowworms*num_glowworms];
        cluster[0] =  sorted_glowworms[0].id; //第0个类别id
        int cluster_index = 1;
        // 每一个类别的数量
        int * cluster_len_idnex = new int [num_glowworms];
        memset(cluster_len_idnex, 0, sizeof(int)*num_glowworms);
        cluster_len_idnex[0]++;
        // 聚类
        
        // 对每一个cluster
        for(int g = 0; g<num_glowworms; g++){
            double * backbone_clusters = new double [receptor.num_atoms*3 + lignad.num_atoms*3];
            double * backbone_glowworms = new double [receptor.num_atoms*3 + lignad.num_atoms*3];
            
            
            int current_g = sorted_glowworms[g].id;
            
            // copy保存的坐标 当前glowworm的坐标
            double * g_coord_rec = &rec_copy[current_g*receptor.num_atoms*3];
            double * g_coord_lig = &lig_copy[current_g*lignad.num_atoms*3];
            bool in_cluster = false;
            for(int c = 0; c<cluster_index; c++){
                int current_cluster = cluster[c*num_glowworms]; 
                //当前cluster取一个代表
                double * cluster_coord_rec = &rec_copy[current_cluster*receptor.num_atoms*3];
                double * cluster_coord_lig = &lig_copy[current_cluster*lignad.num_atoms*3];
                int glowworms_index = 0;
                int cluster_len = 0;
                // receptor
                // 取backbone
                for(int dd = 0; dd<receptor.num_atoms; dd++){
                    // std::cout<<receptor.atom_name[i];
                    if (receptor.atom_name[dd] == "CA" || receptor.atom_name[dd] == "P")
                    {
                        backbone_clusters[cluster_len*3] = cluster_coord_rec[dd*3];
                        backbone_clusters[cluster_len*3+1] = cluster_coord_rec[dd*3+1];
                        backbone_clusters[cluster_len*3+2] = cluster_coord_rec[dd*3+2];

                        backbone_glowworms[glowworms_index*3] = g_coord_rec[dd*3];
                        backbone_glowworms[glowworms_index*3+1] = g_coord_rec[dd*3+1];
                        backbone_glowworms[glowworms_index*3+2] = g_coord_rec[dd*3+2];
                        cluster_len+=1;
                        glowworms_index+=1;
                    }
                    
                }

                for(int dd = 0; dd<lignad.num_atoms; dd++){
                    // std::cout<<lignad.atom_name[i];
                    if (lignad.atom_name[dd] == "CA" || lignad.atom_name[dd] == "P")
                    {
                        backbone_clusters[cluster_len*3] = cluster_coord_lig[dd*3];
                        backbone_clusters[cluster_len*3+1] = cluster_coord_lig[dd*3+1];
                        backbone_clusters[cluster_len*3+2] = cluster_coord_lig[dd*3+2];

                        backbone_glowworms[glowworms_index*3] = g_coord_lig[dd*3];
                        backbone_glowworms[glowworms_index*3+1] = g_coord_lig[dd*3+1];
                        backbone_glowworms[glowworms_index*3+2] = g_coord_lig[dd*3+2];
                        cluster_len+=1;
                        glowworms_index+=1;
                    }
                    
                }
                // cal RMSD
                double rmsd = cal_rmsd(backbone_clusters, backbone_glowworms, cluster_len, glowworms_index);
                if (rmsd <= 4.0){
                    // cluster[cluster_index] = glowworms[g].id;
                    // 当前聚类的长度
                    int cluster_len = cluster_len_idnex[c];
                    
                    cluster[c*num_glowworms+cluster_len] = current_g;
                    cluster_len_idnex[c] ++;
                    in_cluster = true;
                    break;
                }
            }
            
            if(!in_cluster){

                cluster[cluster_index*num_glowworms] = current_g;
                cluster_len_idnex[cluster_index] ++;
                cluster_index ++;
            }

        }

        // write info

        string save_swarm = "swarm_";
        string file_name = "gso_";
        string swarms = to_string(i);
        string steps = to_string(STEPS) ;
        string file_type = ".out";
        string save_dir =  save_swarm+swarms;
        string save_file = file_name+steps+file_type;
        // std::cout<<save_file<<std::endl;
        string slash = "/";
        string full_path = save_dir+slash+save_file;
        string filename = "cluster.repr";
        string cluster_file = save_dir+slash+filename;
        // std::cout<<full_path<<std::endl;
        
        std::ofstream c_fileout(cluster_file.c_str());

        if (c_fileout.is_open()) {
            // 写入文件内容
            for(int c = 0 ; c< cluster_index; c++){
                int cluster_len = cluster_len_idnex[c];
                int id = cluster[c*num_glowworms];
                double scoring = scoring_base[id];
                
                c_fileout << c<<":"<<cluster_len<<":"<<scoring<<":"<<id<<":lightdock_"<<id<<".pdb\n";

            }

        }
            

        // 关闭文件
        c_fileout.close();


        delete [] cluster;
        delete [] cluster_len_idnex;
        

        
    }
    
    
    delete [] prob_array;
    // delete [] glowworms;
    delete [] receptor_pose;
    delete [] ligand_pose;
    delete [] ligand_reference_pose;
    delete [] rec_copy_base;
    delete [] lig_copy_base;
    delete [] select_base;
    delete [] select_postition_base;
    delete [] delta_base;

    delete [] id_base;
    delete [] scoring_base;
    delete [] luciferin_base;
    delete [] rho_base;
    delete [] gamma_base;
    delete [] beta_base;
    delete [] vision_range_base;
    delete [] max_vision_range_base;
    delete [] step_base;
    delete [] moved_base;
    delete [] max_neighbors_base;
    delete [] nnei_len_base;
    delete [] neighbors_base;

}


void prepare_gso_tasks(
    SwarmCenters & centers,Complex & receptor, Complex &lignad, FastDifire & fastdifire, int seed, double step_translation, double step_rotation, bool use_anm, double nmodes_step,
    int anm_rec, int anm_lig, bool local_minimization, int swarms, int num_glowworms, int rank, int size
){
    /*
    each process executes the program
    */
    bool use_cuda = get_env_cuda();
    if (use_cuda && rank==0)
    {
        std::cout<<"use cuda"<<std::endl;
    }
    
    // std::cout<<complex.structure_size[0];
    // std::cout<<complex.chain.size();
    // TODO: chain
    int object_size_rec = receptor.chain.size() * receptor.residue_atom_number[receptor.num_res_type];
    // std::cout<<object_size<<std::endl;
    int * dfire_objects_rec = new int[object_size_rec];
    int num_threads = get_env_num_threads();

    int object_size_lig = lignad.chain.size() * lignad.residue_atom_number[lignad.num_res_type];
    int * dfire_objects_lig = new int[object_size_lig];
    // 加载dfire得分 此处计算正确
    fastdifire.difire_energy = new double [168*168*20];
    string para_file = PROJECT_PATH + string("/data/DCparams");
    std::ifstream ifs(para_file);
    if (!ifs.is_open()) {
        std::cerr << "Error opening file: DCparams"  << std::endl;
        exit(0);
    }
    std::string line;
    int count = 0;
    while (std::getline(ifs, line)) {
        double value = std::stod(line);
        fastdifire.difire_energy[count] = value;

        count++;
    }
    
    
    get_docking_model(dfire_objects_rec, receptor,fastdifire);
    get_docking_model(dfire_objects_lig, lignad, fastdifire);
    
    
    double precision = 0.01;
    double * receptor_reference_points = new double [3];
    double * receptor_poles = new double [3*6];
    double * ligand_reference_points = new double [3];
    double * ligand_poles = new double [3*6];
    minimum_volume_ellipsoid(receptor_reference_points,receptor_poles,receptor.atom_coordinates,precision,receptor.num_atoms);
    minimum_volume_ellipsoid(ligand_reference_points,ligand_poles,lignad.atom_coordinates,precision,lignad.num_atoms);
    // if (use_cuda && rank==0)
    // {
    //     std::cout<<"use cuda"<<std::endl;
    // }
    
    /*
    spwan to each process
    */
    #pragma omp parallel for num_threads(num_threads)
    for(int i = centers.swarm_start; i<centers.swarm_start+centers.swarm_chunk; i++){
        // save
        // string save_dirt = "test/";
        string save_swarm = "swarm_";
        string file_name = "gso_";
        string swarms = to_string(i);
        string steps = to_string(STEPS) ;
        string file_type = ".out";
        string save_dir =  save_swarm+swarms;
        string save_file = file_name+steps+file_type;
        // std::cout<<save_file<<std::endl;
        string slash = "/";
        string full_path = save_dir+slash+save_file;
        // std::cout<<full_path<<std::endl;
        // 创建目录
        if (!fs::exists(save_dir)) {
            fs::create_directory(save_dir);
        }
    }
    
    if(use_cuda){
        
        // TODO:
        cal_gso_tasks_gpu(centers, receptor, lignad, fastdifire, seed, step_translation, step_rotation, use_anm, nmodes_step,
     anm_rec, anm_lig, local_minimization, swarms, num_glowworms, receptor_reference_points, receptor_poles, ligand_reference_points, 
     ligand_poles, dfire_objects_rec, dfire_objects_lig, object_size_rec, object_size_lig, rank, size);
    }else{
        cal_gso_tasks_cpu(centers, receptor, lignad, fastdifire, seed, step_translation, step_rotation, use_anm, nmodes_step,
     anm_rec, anm_lig, local_minimization, swarms, num_glowworms, receptor_reference_points, receptor_poles, ligand_reference_points, 
     ligand_poles, dfire_objects_rec, dfire_objects_lig);
    }
    

    delete [] dfire_objects_rec;
    delete [] dfire_objects_lig;
    delete [] fastdifire.difire_energy;
    delete [] receptor_reference_points;
    delete [] receptor_poles;
    delete [] ligand_reference_points;
    delete [] ligand_poles;

    // printf("finisheddd. \n");
    // free_fastdfire(fastdifire);
    // printf("finisheddd. \n");
}

double cal_rmsd(double * backbone_clusters, double * backbone_glowworm, int cluster_size, int glowworm_size){
    if(cluster_size != glowworm_size){
        std::cout<<"error:"<<cluster_size<<glowworm_size<<std::endl;
    }
    double rmsd = 0;
    for(int i = 0; i<cluster_size; i++){
        rmsd+=(backbone_clusters[i*3] - backbone_glowworm[i*3]) * (backbone_clusters[i*3] - backbone_glowworm[i*3]);
        rmsd+=(backbone_clusters[i*3+1] - backbone_glowworm[i*3+1]) * (backbone_clusters[i*3+1] - backbone_glowworm[i*3+1]);
        rmsd+=(backbone_clusters[i*3+2] - backbone_glowworm[i*3+2]) * (backbone_clusters[i*3+2] - backbone_glowworm[i*3+2]);
    }
    rmsd /= cluster_size;
    rmsd = sqrt(rmsd);
    return rmsd;
}

bool compare(Glowworm g1, Glowworm g2){
    return g1.scoring>g2.scoring;
}



void write_pdb(Complex structure, std::ofstream &newFile, double * rec_copy){
    if (newFile.is_open()) {   
        // string file_buffer =  ""     
        for(int nn = 0; nn<structure.num_atoms; nn++){
            string type = structure.atoms[nn].type;
            string number = to_string(nn+1);
            number = string(5 - number.length(), ' ') + number;
            string atom_name = structure.atoms[nn].name;
            if(atom_name.length()<4){
                atom_name = " "+atom_name;
            }
            // std::cout<<receptor.atoms[nn].alternative<<1212<<std::endl;
            atom_name = atom_name+string(4 - atom_name.length(), ' ');
            string alternative = structure.atoms[nn].alternative + string(1 - structure.atoms[nn].alternative.length(), ' ');

            string residue_name = structure.atoms[nn].residue_name;
            residue_name = string(3 - residue_name.length(),' ') + residue_name ;
            string chain_id = structure.atoms[nn].chain_id;
            chain_id = string(2 - chain_id.length(), ' ')+chain_id;

            string residue_number = to_string(structure.atoms[nn].residue_number);
            residue_number = string(4 - residue_number.length(),' ') + residue_number ;
            string residue_insertion = structure.atoms[nn].residue_insertion;
            residue_insertion = string(1 - residue_insertion.length(),' ') + residue_insertion ;

            string blank = "   ";

            double xx = rec_copy[nn*3];
            double yy = rec_copy[nn*3+1];
            double zz = rec_copy[nn*3+2];
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << xx;  // 设置小数点精度为3位
            string xx_str = ss.str();
            xx_str = string(8 - xx_str.length(),' ') + xx_str ;

            std::stringstream ss_y;
            ss_y << std::fixed << std::setprecision(3) << yy;  // 设置小数点精度为3位
            string yy_str = ss_y.str();
            yy_str = string(8 - yy_str.length(),' ') + yy_str ;

            std::stringstream ss_z;
            ss_z << std::fixed << std::setprecision(3) << zz;  // 设置小数点精度为3位
            string zz_str = ss_z.str();
            zz_str = string(8 - zz_str.length(),' ') + zz_str ;

            double occupancy = structure.atoms[nn].occupancy;
            std::stringstream s_o;
            s_o << std::fixed << std::setprecision(2) << occupancy;
            string oo = s_o.str();
            oo = string(6 - oo.length(),' ') + oo ;

            double b_factor = structure.atoms[nn].b_factor;
            std::stringstream s_b;
            s_b << std::fixed << std::setprecision(2) << b_factor;
            string bb = s_b.str();
            bb = string(6 - bb.length(),' ') + bb ;

            string element = structure.atoms[nn].element;
            element = string(12 - element.length(),' ') + element ;
            // TODO: 重复
            
            // zz_str = string(8 - zz_str.length(),' ') + zz_str ;

            // string residue_name = receptor.atoms[nn].residue_name;
            newFile << type + number + " "+atom_name+ alternative+ residue_name+chain_id
                + residue_number+residue_insertion+blank+xx_str+yy_str+zz_str+oo+bb+element+"\n";
        }       
    // 写入文件内容
    // newFile << "#Coordinates  RecID  LigID  Luciferin  Neighbor's number  Vision Range  Scoring\n";
    }
}


void move_extent(double * self, double * other, int num_ext,double step_nmodes){
    double * delta_x = new double [num_ext];
    double n = 0;
    for(int i = 0; i< num_ext; i++){
        delta_x[i] = other[i] - self[i];
        n += delta_x[i] * delta_x[i];
    }   
    
    n = sqrt(n);
    if(n>=1e-8){
        for(int i = 0; i< num_ext; i++){
            delta_x[i] *= step_nmodes/n;
            self[i] += delta_x[i];
        }
    }
    delete [] delta_x;
}
// TODO result
void slerp(double * self, double * other, double rotation_step){
    // calculate inter polation
    double self_norm2 = 0;
    double other_norm2 = 0;

    for(int i = 0; i<4;i++){
        self_norm2 += (self[i] * self[i]);
        other_norm2 += (other[i] * other[i]);
    }
    self_norm2 = sqrt(self_norm2);
    other_norm2 = sqrt(other_norm2);
    for(int i = 0; i<4;i++){
        self[i] /= self_norm2;
        other[i] /= other_norm2;
    }
    // calc dot
    double q_dot = 0;
    for(int i = 0; i<4;i++){
        q_dot += self[i] * other[i];
    }
    if(q_dot < 0){
        q_dot = q_dot * -1;
        for(int i = 0; i< 4; i++){
            self[i] = -self[i];
        }
    }

    if(q_dot>0.9995){
        for(int i = 0; i < 4; i++){
            self[i] = self[i] + rotation_step * (other[i] - self[i]);
        }
        double result_norm = 0;
        for(int i = 0; i< 4; i++){
            result_norm += (self[i] * self[i]);
        }
        result_norm = sqrt(result_norm);
        for(int i = 0; i<4;i++){
            self[i] /= result_norm;
        }
    }else{
        q_dot = (q_dot < 1) ? q_dot : 1;
        q_dot = (q_dot > -1) ? q_dot : -1;
        double omega = acos(q_dot);
        double so = sin(omega);
        double tmp0 = (sin((1.0-rotation_step)*omega) / so);
        double tmp1 = (sin(rotation_step*omega)/so);
        for(int i = 0; i< 4; i++){
            self[i] = tmp0 * self[i] + tmp1 * other[i];
        }
    }

}


int select_random_neighbor(
    int * select, double prob, int * neighbors,
    double * probabilities,int select_index,int n, int nnei_len, int swarm
){
    /*
    select_index： 选择索引
    neighbors： 邻居表

    */
    // n: current glowworm
    // TODO: issue
    // int nnei_len = nnei_len; //获取邻居数量
    // 如果没有邻居，索引是自己
    if (nnei_len == 0){
        select[n] = n;
    // 否则，索引是根据概率得到的
    }else{
        double sum_probabilities = 0.0;
        int ii = 0;
        while (sum_probabilities < prob)
        {
            sum_probabilities+=probabilities[ii];
            ii+=1;
        }
        // 得到第几个neighbor
        
        select[n] = neighbors[ii-1];
        if(swarm==0 && n==56){
            printf("idx %d\n",neighbors[ii-1]);
        }
        
    }

    return select_index;
    
}



void compute_probability_moving_from_neighbors(
    double * probabilities, int * neighbors, double * luciferin, int nnei_len, double current_luciferin
){
    /*
    该方法目前存疑
    probabilities: 概率
    neighbors： 对应swarm ， 对应glowwormd的 邻居表
    luciferin： 当前swarm的luciferin
    nnei_len：当前 swarm 当前glowworm的邻居表长度
    current_luciferin：当前原子的luciferin
    */
    double total_sum = 0;
    // double * probabilities = new double [current_glowworm->nnei_len];
    for(int idx = 0; idx<nnei_len; idx++){
        int nei_id = neighbors[idx];
        // double current_luciferin = luciferin[idx];
        double neighbor_luciferin = luciferin[nei_id];
        double difference = neighbor_luciferin - current_luciferin;
        probabilities[idx] = difference;
        total_sum += difference;

    }
    for(int idx = 0; idx<nnei_len; idx++){
            probabilities[idx] /= total_sum;
    }
}

void euclidean_dist(
    unsigned int * indexes, int * indexes_len, double * receptor_coordinate, 
    double * ligand_coordinate, int rec_len, int lig_len){
    // unsigned int i, j;
    *indexes_len = 0;
    unsigned int n = 0;
    // int count = 0;
    // *indexes = malloc(3*rec_len*lig_len*sizeof(unsigned int));
    // std::cout<<rec_len<<std::endl;
    // std::cout<<lig_len<<std::endl;
    // shooting !!!!!!!!!!!!!!!!!!
    for (int i = 0; i < rec_len; i++) {
        for (int j = 0; j < lig_len; j++) {
        double dist = (receptor_coordinate[i*3+0] - ligand_coordinate[j*3+0]) * (receptor_coordinate[i*3+0] - ligand_coordinate[j*3+0]) +
                      (receptor_coordinate[i*3+1] - ligand_coordinate[j*3+1]) * (receptor_coordinate[i*3+1] - ligand_coordinate[j*3+1]) +
                      (receptor_coordinate[i*3+2] - ligand_coordinate[j*3+2]) * (receptor_coordinate[i*3+2] - ligand_coordinate[j*3+2]);
            
            if (dist <= 225.) {
                indexes[n++] = i;
                indexes[n++] = j;
                // distance[n++] = (sqrt(dist)*2.0 - 1.0);
                indexes[n++] = (sqrt(dist)*2.0 - 1.0);
                (*indexes_len)++;

            }
        }
    }

}


// other optimized
void cdfire_calculate_dfire(
    double * energy, int indexes_len,
    int * interface_receptor, int * interface_ligand,  int * rec_objects, 
    int * lig_objects, FastDifire & fastdifire, double * receptor_coordinate, double * ligand_coordinate, 
    int rec_len, int lig_len, int num_glowworms,int swarms){
    int n, m, i, j, dfire_bin, atoma, atomb,  interface_len,d;
    double interface_cutoff, *dfire_en_array;
    // double d;
    interface_cutoff = 3.9;
    // energy = 0.;
    interface_len = 0;
    // unsigned int * indexes = new unsigned int[3*rec_len*lig_len];
    // euclidean_dist(indexes, indexes_len, receptor_coordinate, ligand_coordinate, rec_len, lig_len);

    indexes_len = 0;
    unsigned int nn = 0;

    for (int i = 0; i < rec_len; i++) {
        double v = 0;
        for (int j = 0; j < lig_len; j++) {
        double dist = (receptor_coordinate[i*3+0] - ligand_coordinate[j*3+0]) * (receptor_coordinate[i*3+0] - ligand_coordinate[j*3+0]) +
                      (receptor_coordinate[i*3+1] - ligand_coordinate[j*3+1]) * (receptor_coordinate[i*3+1] - ligand_coordinate[j*3+1]) +
                      (receptor_coordinate[i*3+2] - ligand_coordinate[j*3+2]) * (receptor_coordinate[i*3+2] - ligand_coordinate[j*3+2]);
            
            if (dist <= 225.) {
                // index 有变化
                // indexes[nn++] = i;
                // indexes[nn++] = j;
                // // distance[n++] = (sqrt(dist)*2.0 - 1.0);
                // indexes[nn++] = (sqrt(dist)*2.0 - 1.0);

                d = (sqrt(dist)*2.0 - 1.0);
                atoma = rec_objects[i];
                atomb = lig_objects[j];
                dfire_bin = dist_to_bins[d] - 1;
                unsigned int array_ = atoma*168*20 + atomb*20 + dfire_bin;
                double value = fastdifire.difire_energy[array_];
                v += value;
                (indexes_len)++;

            }
        }
        *energy += v;
        if(num_glowworms == 3 && swarms==3 && i == 3841){
            printf("%f %f %f \n", receptor_coordinate[i*3+0],receptor_coordinate[i*3+1],receptor_coordinate[i*3+2]);
            printf("test %f \n",v);
        }
        
    }
    *energy = ((*energy)*0.0157 - 4.7)*-1;


    // unsigned int * array = new unsigned int [(indexes_len)];
    // // interface_receptor = malloc((indexes_len)*sizeof(int));
    // // interface_ligand = malloc((indexes_len)*sizeof(int));

    // for (n = m = 0; n < (indexes_len); n++) {

    //     i = indexes[m++];
    //     j = indexes[m++];
    //     d = indexes[m++];

    //     // if (d <= interface_cutoff) {
    //     //     // std::cout<<"prime0"<<std::endl;
    //     //     interface_receptor[interface_len] = i;
    //     //     interface_ligand[interface_len++] = j;

    //     // }

    //     atoma = rec_objects[i];
    //     atomb = lig_objects[j];
    //     // std::cout<<"prime1"<<std::endl;
    //     dfire_bin = dist_to_bins[d] - 1;
    //     // std::cout<<"prime2"<<std::endl;
    //     array[n] = atoma*168*20 + atomb*20 + dfire_bin;
    //     // std::cout<<"n:"<<n<<std::endl;
    // }
    // // std::cout<<indexes_len<<std::endl;
    // // 获取array index
    // for(i = 0; i<(indexes_len); i++){
    //     double value = fastdifire.difire_energy[array[i]];
    //     *energy += value;
    // }
    // *energy = ((*energy)*0.0157 - 4.7)*-1;
    // free(array);
    // delete [] array;
    // delete [] indexes;
    // delete [] distance;
    // free(interface_receptor);
    // free(interface_ligand);
    // std::cout<<energy<<std::endl;

    // return 
}



void get_default_box(double *bounding_box, int anm_rec, int anm_lig, bool use_anm){
    int b = 0; 
    for(; b<3; b++){
        bounding_box[b*2] = -MAX_TRANSLATION;
        bounding_box[b*2+1] = MAX_TRANSLATION;
    }
    for (;b<7;b++){
        bounding_box[b*2] = -MAX_ROTATION;
        bounding_box[b*2+1] = MAX_ROTATION;
    }
    if(use_anm){
        for(; b<7+anm_lig;b++){
            bounding_box[b*2] = MIN_EXTENT;
            bounding_box[b*2+1] = MAX_EXTENT;
        }
        for(; b<7+anm_lig+anm_rec;b++){
            bounding_box[b*2] = MIN_EXTENT;
            bounding_box[b*2+1] = MAX_EXTENT;
        }
    }
}

