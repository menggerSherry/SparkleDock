#include "cal_anm.h"
#include "lib_tools.h"
#include <Eigen/Dense>
#include <mpi.h>
// #include <cublas_v2.h>
#include <fstream>
#include <cmath>
#include <random>
#include <time.h>
#include "freesasa.h"
#include "kmeans.h"
#include "swarm_centers.h"
#include <iomanip>  // 包含设置小数点精度的头文件
#define DEFAULT_EXTENT_MU 0.0

#define DEFAULT_EXTENT_SIGMA 1.0

// #define DEFAULT_NUM_SWARMS  400

// #define DEFAULT_NUM_GLOWWORMS  200

#define DEFAULT_SWARM_RADIUS 10.0

#define DEFAULT_SPHERES_PER_CENTROID 100

#define PI 3.14159265358979323846

#define SWARM_DISTANCE_TO_SURFACE_CUTOFF 3.0

static int num_threads = get_env_num_threads();


void eigh_eigen(double* eigenvalues, double* eigenvectors, double* matrix, int* eigvals, int n) {
    // Convert input matrix to Eigen format
    Eigen::MatrixXd mat = Eigen::Map<Eigen::MatrixXd>(matrix, n, n);

    // Perform Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mat);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Eigen decomposition failed!" << std::endl;
        return;
    }

    // Extract eigenvalues and eigenvectors
    Eigen::VectorXd values = solver.eigenvalues();
    Eigen::MatrixXd vectors = solver.eigenvectors();

    int start = eigvals[0];
    int end = eigvals[1];
    int length = end - start + 1;

    for (int i = 0; i < length; ++i) {
        eigenvalues[i] = values[start + i];
        for (int j = 0; j < n; ++j) {
            eigenvectors[j * length + i] = vectors(j, start + i);
        }
    }
}

void cal_anm(Complex & complex, ANM & anm, int n_modes, double rmsd, int seed, int swarms, int glowworms, int rank, int size){
    
    norm(complex.atom_coordinates,complex.num_atoms);
    vector<double> backbone_coord_buff;
    vector<int> backbone_index;
    string backbone_name1 = "CA";
    string backbone_name2 = "C4";
    int backbone_size = 0;
    select_atom_by_name(backbone_coord_buff,backbone_index, complex.atom_coordinates, complex.atom_name, backbone_name1);
    backbone_size = backbone_coord_buff.size();
    if(backbone_size == 0){
        select_atom_by_name(backbone_coord_buff,backbone_index, complex.atom_coordinates, complex.atom_name, backbone_name2);
        backbone_size = backbone_coord_buff.size();
    }
    anm.backbone_coord = new double[backbone_size];
    memcpy(anm.backbone_coord,&backbone_coord_buff[0], sizeof(double)*backbone_coord_buff.size());
    // print_array(anm.backbone_coord, backbone_size);

    double cutoff = 15;
    double gamma = 1.;
    int n_atoms = backbone_size / 3;
    anm.dof = backbone_size;
    // double ** hessian = new double * [backbone_size];
    anm.kirchhoff = new double [n_atoms * n_atoms];

    anm.hessian = new double [backbone_size * backbone_size];
    memset(anm.hessian, 0.0, backbone_size * backbone_size*sizeof(double));
    memset(anm.kirchhoff, 0.0, n_atoms * n_atoms*sizeof(double));
    build_hessian1D(anm.hessian,anm.kirchhoff, cutoff, gamma, n_atoms, anm.backbone_coord);
    bool zeros = false;
    bool turbo = true;
    int expct_n_zeros = 6;
    bool reverse = false;
    calModes(anm,n_modes,zeros,turbo,expct_n_zeros,reverse,n_atoms, rank, size);
    anm.trace = 0;
    // std::cout<<anm.n_modes
    for(int i = 0; i<anm.n_modes;i++){
        anm.trace += anm.invvals[i];
    }
    if (rank == 0)
    {
        std::cout<<"trace"<<anm.trace<<std::endl;
    }
    vector <string> residues;

    int * res_real_indices = new int [complex.num_atoms];
    int * indices = new int [complex.num_atoms*3];
    memset(res_real_indices,-1, sizeof(int)*complex.num_atoms);
    #pragma omp parallel for num_threads(num_threads)
    for(int i = complex.residue_size[0]; i< complex.residue_size[1]; i++){
        bool fastin = false;

        for(int res_num = complex.residue_atom_number[i]; res_num<complex.residue_atom_number[i+1];res_num++){
            // int res_tmp = complex.residue_atom_number[res_num];
            if(res_num == complex.residue_atom_number[i]+1){
                res_real_indices[res_num] = i;
                
            }
            // std::cout<<res_real_indices[res_num]<<" ";
            for(int d = 0; d < 3; d++){
                indices[res_num*3+d] = i*3+d;
                
            }
        }
    }
    // print_array(indices,complex.num_atoms*3 );
    // step2 res_real_indices
    #pragma omp parallel for num_threads(num_threads)
    for(int i = complex.residue_size[0]; i< complex.residue_size[1]; i++ ){
        int sum_ = 0;
        for(int res_num = complex.residue_atom_number[i]; res_num<complex.residue_atom_number[i+1];res_num++ ){
            if(res_real_indices[res_num]>=0){
                sum_+=1;
            }
        }
        if(sum_==1) continue;
        else{
            std::cout<<"developing"<<std::endl;
        }
        
    }
    // finally get the indices
    // vec
    // evec try to compute 
    // std::cout<<complex.num_atoms*3<<std::endl;
    double * evec_buffer = new double[complex.num_atoms*3*anm.n_modes];
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<complex.num_atoms*3; i++){
        int index = indices[i];
        memcpy(&evec_buffer[i*anm.n_modes], &anm.eigvecs[index*anm.n_modes],sizeof(double)*anm.n_modes);
    }
    
    // exit(-1);
    // 得到标准化的evec矩阵。
    #pragma omp parallel for num_threads(num_threads)
    for(int val = 0; val < anm.n_modes; val++){
        double sum_ = 0;
        for(int ii = 0; ii<complex.num_atoms*3;ii++){
            sum_+=pow(evec_buffer[ii*anm.n_modes+val],2);
        }
        sum_ = sqrt(sum_);
        for(int ii = 0; ii<complex.num_atoms*3;ii++){
            evec_buffer[ii*anm.n_modes+val] /= sum_;
        }
    }
    
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(DEFAULT_EXTENT_MU, DEFAULT_EXTENT_SIGMA);
    
    double coef = 0;

    unsigned int seeds = 324324;
    generator.seed(seeds);
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<swarms*glowworms; i++){
        double sum_ = 0;
        for(int j = 0; j<anm.n_modes; j++){
            double value = distribution(generator);
            double var = 1.0/anm.eigvals[j];
            sum_ +=  value * value * var ;
        }
        sum_ = sqrt(sum_);
        coef+=sum_;
    }
    coef =  coef/ (swarms*glowworms);
    
    double scale_item = sqrt(complex.structure_size[1]) * rmsd /coef;
    
    double * scale = new double[anm.n_modes];
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i < anm.n_modes; i++){
        // std::cout<<1.0/anm.eigvals[i]<<std::endl;
        scale[i] = scale_item * sqrt(1.0/anm.eigvals[i]);
    }
    // print_array(scale,anm.n_modes);
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i< anm.n_modes; i++){
        for(int j = 0; j<complex.num_atoms*3; j++){
            
            evec_buffer[j*anm.n_modes+i]  *= scale[i];
        }
    }
    complex.n_modes = anm.n_modes;
    complex.modes = new double[complex.num_atoms*3*anm.n_modes];
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i< anm.n_modes; i++){
        for(int j = 0; j<complex.num_atoms*3; j++){
            complex.modes[i*complex.num_atoms*3+j] = evec_buffer[j*anm.n_modes+i];
        }
    }
    
    delete [] evec_buffer;
    delete [] res_real_indices;
    delete [] indices;
    delete [] scale;

    // print_mat(complex.modes,anm.n_modes,complex.num_atoms*3);

}

void free_anm(ANM & anm){
    delete [] anm.backbone_coord;
    delete [] anm.kirchhoff;
    delete [] anm.hessian;
    delete [] anm.eigvals;
    delete [] anm.eigvecs;
    delete [] anm.invvals;
}

void select_atom_by_name(vector<double> & backbone_coord, vector<int> & backbone_index, double * coord, vector<string> names, string atom_name){
    
    for(int i = 0; i<names.size(); i++){
        if(names[i] == atom_name){
            backbone_coord.push_back(coord[i*3]);
            backbone_coord.push_back(coord[i*3+1]);
            backbone_coord.push_back(coord[i*3+2]);
            backbone_index.push_back(i);
        }

    }
    
}


void build_hessian(double ** hessian,double * kirchhoff, double cutoff, double gamma, int n_atoms, double *coords){
    double cutoff2 = cutoff * cutoff;
    int dof = n_atoms *3;
    
    int count = 0;
    // double * hessian = new double [dof*dof];
    for(int i = 0; i< n_atoms; i++){
        int res_i3 = i*3;
        int res_i33 = res_i3+3;
        int i_p1 = i+1;
        // cal i2j
        double * coord_i = &coords[i*3];
        
        count++;
        // std::cout<<count<<std::endl;
        // int j_index = 0;
        for(int j = i_p1; j<n_atoms; j++){
            int res_j3 = j * 3;
            int res_j33 = res_j3 + 3;

            double i2j_all[3] = {coords[j * 3] - coords[i * 3],
                                 coords[j * 3 + 1] - coords[i * 3 + 1],
                                 coords[j * 3 + 2] - coords[i * 3 + 2]};

            double dist2 = 0.0;
            for (int k = 0; k < 3; k++) {
                dist2 += i2j_all[k] * i2j_all[k];
            }

            if (dist2 > cutoff2) {
                continue;
            }

            double g = 1; // Example gamma function, you need to define your own

            double super_element[3][3];
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    super_element[k][l] = i2j_all[k] * i2j_all[l] * (-g / dist2);
                }
            }
            // int row, col;


            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    hessian[res_i3 + k][res_j3 + l] = super_element[k][l];
                    hessian[res_j3 + l][res_i3 + k] = super_element[k][l];
                    hessian[res_i3 + k][res_i3 + l] -= super_element[k][l];
                    hessian[res_j3 + k][res_j3 + l] -= super_element[k][l];
                }
            }

            kirchhoff[i * n_atoms + j] = -g;
            kirchhoff[j * n_atoms + i] = -g;
            kirchhoff[i * n_atoms + i] += g;
            kirchhoff[j * n_atoms + j] += g;

        }
    }
}

void build_hessian1D(double *hessian, double *kirchhoff, double cutoff, double gamma, int n_atoms, double *coords){
    double cutoff2 = cutoff * cutoff;
    int dof = n_atoms * 3;
    // #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n_atoms; i++) {
        int res_i3 = i * 3;
        int i_p1 = i + 1;
        double *coord_i = &coords[i * 3];

        for (int j = i_p1; j < n_atoms; j++) {
            int res_j3 = j * 3;

            double i2j_all[3] = {
                coords[j * 3] - coord_i[0],
                coords[j * 3 + 1] - coord_i[1],
                coords[j * 3 + 2] - coord_i[2]
            };

            double dist2 = 0.0;
            for (int k = 0; k < 3; k++) {
                dist2 += i2j_all[k] * i2j_all[k];
            }

            if (dist2 > cutoff2) {
                continue;
            }

            double g = gamma;

            double super_element[3][3];
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    super_element[k][l] = i2j_all[k] * i2j_all[l] * (-g / dist2);
                }
            }

            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    // 压缩 Hessian 矩阵为一维数组
                    hessian[(res_i3 + k) * dof + (res_j3 + l)] = super_element[k][l];
                    hessian[(res_j3 + l) * dof + (res_i3 + k)] = super_element[k][l];
                    hessian[(res_i3 + k) * dof + (res_i3 + l)] -= super_element[k][l];
                    hessian[(res_j3 + l) * dof + (res_j3 + k)] -= super_element[k][l];
                }
            }

            kirchhoff[i * n_atoms + j] = -g;
            kirchhoff[j * n_atoms + i] = -g;
            kirchhoff[i * n_atoms + i] += g;
            kirchhoff[j * n_atoms + j] += g;
        }
    }
}


void eigh_cpu(double *eigenvalues, 
    double *eigenvectors, 
    const double *matrix, 
    const int *eigvals, 
    int n, 
    int rank, 
    int size) {

    // 将 matrix 转换为 Eigen 库的 MatrixXd 格式
    Eigen::Map<const Eigen::MatrixXd> mat(matrix, n, n);

    // 创建自伴特征值分解的求解器
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(mat);

    // 检查求解是否成功
    if (solver.info() != Eigen::Success) {
    std::cerr << "Eigen decomposition failed!" << std::endl;
    exit(EXIT_FAILURE);
    }

    // 获取特征值和特征向量
    Eigen::VectorXd eig_vals = solver.eigenvalues().real();  // 仅实部
    Eigen::MatrixXd eig_vecs = solver.eigenvectors().real(); // 仅实部

    // 根据 eigvals 范围提取特征值和特征向量
    int start = eigvals[0], end = eigvals[1], length = end - start + 1;

    // 提取特征值
    for (int i = 0; i < length; ++i) {
    eigenvalues[i] = eig_vals[start + i];
    }

    // 提取特征向量
    for (int i = 0; i < n; ++i) {
    for (int j = 0; j < length; ++j) {
    eigenvectors[i * length + j] = eig_vecs(i, start + j);
    }
    }
}


void calModes(ANM  &anm, int n_modes, bool zeros, bool turbo, int expct_n_zeros, bool reverse, int n_atoms, int rank, int size){
    /*
    anm.eigvals : vals;
    anm.eigvecs : vectors;
    anm.invvals : vars;
    */
    int dof = n_atoms * 3;
    // anm.dof = dof;
    bool warn_zeros;
    int eigvals[2];
    int final_n_zeros;
    if (expct_n_zeros != 0){
        // warn_zeros
        warn_zeros = true;
    }else{
        warn_zeros = false;
    }
    if (n_modes == 0){
        n_modes = dof;
    }else{
        if (reverse){
            eigvals[0] = dof-expct_n_zeros-n_modes;
            eigvals[1] = dof-1;
        }else{
            eigvals[0] = 0;
            eigvals[1] = n_modes+expct_n_zeros-1;
        }
    }
    
    // std::cout<<eigvals[0]<<" "<<eigvals[1]<<std::endl;
    int length = eigvals[1] - eigvals[0] +1;
    double * eigh_vector = new double [dof*length];
    double * eigh_value = new double [length];

    eigh_gpu(eigh_value,eigh_vector,anm.hessian,eigvals,dof, rank, size);

    int n_zeros = 0;
    for(int i = 0; i < length; i++){
        if(eigh_value[i] < 1e-6){
            n_zeros ++ ;
        }
    }

    // std::cout<<n_modes<<":"<<dof<<":"<<n_zeros<<std::endl;

    int val_size;
    
    if(!zeros){
        if(n_zeros > expct_n_zeros){
            if (n_zeros == n_modes + expct_n_zeros && n_modes < dof && rank == 0){
                std::cout<<"zero eigenvalues detected"<<std::endl;
            }
            if(n_modes < dof){
                std::cout<<"out the n modes"<<std::endl;
            }
        }
        // std::cout<<n_zeros<<std::endl;
        final_n_zeros = n_zeros + n_modes;
        anm.n_modes = n_modes;
        anm.dof = dof;
        anm.eigvals = new double [n_modes];
        anm.eigvecs = new double [dof * n_modes];
        
        anm.invvals = new double [n_modes];
        
        // memecpy
        memcpy(anm.eigvals,eigh_value+n_zeros,sizeof(double)*n_modes);
        #pragma omp parallel for num_threads(num_threads)
        for(int i = 0; i < dof; i++){
            for(int j = 0; j< n_modes;j++){
                anm.eigvecs[i*n_modes+j] = eigh_vector[i*length+j+n_zeros];
            }
        }
        #pragma omp parallel for num_threads(num_threads)
        for(int i = 0; i< n_modes;i++){
            anm.invvals[i] = 1/anm.eigvals[i];
        }
        // print_mat(anm.eigvecs, dof, n_modes);

    }else{
        std::cout<<111<<std::endl;
    }
    
    // std::cout<<anm.n_modes<<std::endl;
    if (reverse){
        std::cout<<"reverse"<<std::endl;
        double tmp;
        for(int i = 0; i<anm.n_modes/2; i++){
            tmp = anm.eigvals[i];
            anm.eigvals[i]=anm.eigvals[anm.n_modes-1-i];
            anm.eigvals[anm.n_modes-1-i] = tmp;

            tmp = anm.invvals[i];
            anm.invvals[i] = anm.invvals[anm.n_modes-1-i];
            anm.invvals[anm.n_modes-i-1] =  tmp;

        }
        for(int i = 0;i<dof;i++){
            for(int j = 0; j<anm.n_modes/2;j++){
                tmp = anm.eigvecs[i*anm.n_modes+j];
                anm.eigvecs[i*anm.n_modes+j] = anm.eigvecs[i*anm.n_modes+anm.n_modes-1-j];
                anm.eigvecs[i*anm.n_modes+anm.n_modes-1-j] = tmp;
            }
        }
    }

    delete [] eigh_vector;
    delete [] eigh_value;
    
}



void norm(double *array, int size){
    double mean0 = 0;
    double mean1 = 0;
    double mean2 = 0;
    double sumOfSquares0 = 0;
    double sumOfSquares1 = 0;
    double sumOfSquares2 = 0;
    for(int i = 0; i< size; i++){
        mean0 += array[i*3];
        mean1 += array[i*3+1];
        mean2 += array[i*3+2];
        
    }
    mean0 = mean0/size;
    mean1 = mean1/size;
    mean2 = mean2/size;
    for(int i = 0; i<size; i++){
        array[i*3] -= mean0;
        array[i*3+1] -= mean1;
        array[i*3+2] -= mean2;
    }

}


void calculate_surface_point(
    SwarmCenters & center, 
    Complex & receptor, Complex & ligand, int swarms,int glowworms, int start_points_seed, 
    double * rec_translation,  double * lig_translation,
    double surface_density, bool use_anm, int anm_seed,
    int anm_rec, int anm_lig, bool membrane, bool transmambrane,
    bool write_starting_positions, double swarm_radius, bool flip,
    double fixed_distance, int swarms_per_restraint, bool dense_smapling,  const char * rec_name, int number_of_points, int rank, int size
){
    // int num_threads = get_env_num_threads();
    /* each swarm calculate the swarm number */
    int rec_dis_size = receptor.num_atoms*(receptor.num_atoms-1)/2;
    int ligand_dis_size = ligand.num_atoms*(ligand.num_atoms-1)/2;
    double * distances_matrix_rec = new double [rec_dis_size];
    double * distances_matrix_lig = new double[ligand_dis_size];
    memset(distances_matrix_rec,0.0, sizeof(double)*rec_dis_size);
    memset(distances_matrix_lig, 0.0, sizeof(double)*ligand_dis_size);
    // distance mat
    cal_dist(distances_matrix_rec,receptor.atom_coordinates,receptor.num_atoms,rec_dis_size);
    cal_dist(distances_matrix_lig,ligand.atom_coordinates,ligand.num_atoms,ligand_dis_size);

    double receptor_max_diameter = 0;
    double ligand_max_diameter = 0;
    for(int i = 0; i<rec_dis_size; i++){
        if (receptor_max_diameter<distances_matrix_rec[i]) {receptor_max_diameter = distances_matrix_rec[i];}
    }
    center.receptor_diameter = receptor_max_diameter;
    for(int i = 0; i<ligand_dis_size;i++){
        if(ligand_max_diameter<distances_matrix_lig[i]) {ligand_max_diameter = distances_matrix_lig[i];}
    }
    center.ligand_diameter = ligand_max_diameter;

    // surface distance
    double surface_distance;
    if(ligand_max_diameter<DEFAULT_SWARM_RADIUS*2){
        surface_distance = DEFAULT_SWARM_RADIUS;
    }else{
        surface_distance = ligand_max_diameter / 4.0;
    } 
    double * surface_coord = new double [receptor.structure_size[1]*3];
    int coord_size = 0;
    coord_size = select_surface(surface_coord,receptor);
    if (swarms==0){
        if (membrane){
            std::cout<<"developing"<<std::endl;
        }else{
            // TODO: need optimize
            freesasa_result* result;
            freesasa_structure* structure;
            freesasa_node* node;
            structure = freesasa_structure_new();
            FILE *pdbFile = fopen(rec_name, "r");
            structure = freesasa_structure_from_pdb(pdbFile,NULL,0);
            size_t numAtoms = 0;
            // std::cout<<freesasa_structure_n(structure)<<std::endl;
            fclose(pdbFile);
            result = freesasa_calc_structure(structure, NULL);
            swarms = ceil(result->total / surface_density); //num_of points
            // std::cout<<"num_swarms:"<<swarms<<std::endl;
           
        }
    }
    // correct
    center.swarms = swarms;
    if (rank == 0){
        std::cout<<"num_swarms:"<<swarms<<std::endl;
    }
    
    /* each swarm calculate the swarm number */

    // split according to the swarms
    if(center.swarms < size){
        if (rank == 0){
            std::cerr<<"too many mpi proc"<<std::endl;
        }
        
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* split swarm chunks according to the proc num */
    int base   = center.swarms / size;          // 每个进程的基础分配数量
    int offset = center.swarms % size;          // 前 offset 个进程多分配1个
    int chunk  = (rank < offset) ? (base + 1) : base;
    center.swarm_chunk = chunk;
    center.swarm_start = (rank < offset) ? (rank * (base + 1))
                                        : (offset * (base + 1) + (rank - offset) * base);
    /* split swarm chunks according to the proc num */
    double * surface_centroids = new double[swarms*3];
    int * group = new int[coord_size];
    

    
    /* each swarm calculate the swarm number */
    if(coord_size>swarms){
        // kmeans++
        kmeans_init_centers3(surface_centroids, surface_coord,coord_size,swarms,324324);
        run_kmeans3(group,surface_coord,surface_centroids,100,coord_size,swarms);
        
    }else{
        std::cerr<<"developing swarm size = coord size"<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // print_mat(surface_centroids,swarms,3);
    // Create points over the surface of each surface cluster
    // std::cout<<number_of_points<<std::endl;
    double * sampling = new double [swarms*number_of_points*3];
    memset(sampling, 0.0, sizeof(double)*swarms*number_of_points*3);
    
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<swarms; i++){
        double * current_surface_coo = &surface_centroids[i*3];
        double * current_sampling = &sampling[i*number_of_points*3];
        points_on_sphere(current_sampling,number_of_points);
        // print_mat(current_sampling,number_of_points,3);
        for(int j = 0; j< number_of_points; j++){
            for(int d = 0; d<3; d++){
                current_sampling[j*3+d] = current_sampling[j*3+d] * surface_distance + current_surface_coo[d];
            }
        }
        // 
    }

    // filter out not compatible points
    double * sample_filterd = new double[swarms*number_of_points*3];
    memset(sample_filterd,0.0, sizeof(double)*swarms*number_of_points*3);
    // point数量

    // 找到邻居sample满足的点
    // 满足要求的为1否则为0
    int * indices = new int [swarms*number_of_points];
    memset(indices, -1,sizeof(int)*swarms*number_of_points);
    // 遍历每个sample
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<swarms; i++){
        double * current_centroid = &surface_centroids[i*3];
        double * current_sampling = &sampling[i*number_of_points*3];
        int * current_indices = &indices[i*number_of_points];
        // double * current_sampling_filterd = &sample_filterd[i*number_of_points*3];
        // i满足条件的邻居
        for(int j = 0; j<swarms; j++){
            // get neighbor
            double * neighbor_centorid = &surface_centroids[j*3];
            // find neighbor in radius less than 20
            if(Distance3(current_centroid, neighbor_centorid)<=20.0 && j!=i) {
                for(int p = 0; p<number_of_points; p++){
                    // got value less than surface distance if exist add
                    if(Distance3(&current_sampling[p*3],neighbor_centorid)<=surface_distance){
                        current_indices[p] = 1;
                    }
                }
            }
            
        }

    }
    int index = 0;
    // filter  not compatible points
    // #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i< swarms; i++){
        double * current_sampling = &sampling[i*number_of_points*3];
        for(int j = 0 ;j<number_of_points;j++){
            int current_indices = indices[i*number_of_points+j];
            
            if(current_indices!=1){
                sample_filterd[index*3] = current_sampling[j*3];
                sample_filterd[index*3+1] = current_sampling[j*3+1];
                sample_filterd[index*3+2] = current_sampling[j*3+2];
                index++;
            }
        }
    }
    if (rank==0)
    {
        std::cout<<"Swarms after incompatible filter:"<<index<<std::endl;
    }
    
    // Filter interior points
    double * s = new double [index*3];
    memset(s,0.0, sizeof(double)*index*3);
    int index2 = 0;
    // #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<index;i++){
        double * swam = &sample_filterd[i*3];
        double count = 0;
        // 对总的原子坐标系查找满足条件的邻居坐标
        for(int j = 0; j< receptor.structure_size[1]; j++){
            double * current_structure = &receptor.atom_coordinates[j*3];
            double dis = Distance3(swam,current_structure);
            // some coord in swam radius
            if(dis<=SWARM_DISTANCE_TO_SURFACE_CUTOFF) {count++;}

        }
        if(count == 0){
            s[index2*3] = swam[0];
            s[index2*3+1] = swam[1];
            s[index2*3+2] = swam[2];
            index2++;
        }
    }
    if(rank==0){
        std::cout<<"Swarms after interior points filter: "<<index2<<std::endl;
    }
    
    // final cluster points
    center.s = new double [swarms*3];
    int * s_group = new int[index2];
    if(index2>swarms && !dense_smapling){
        // TODO: random seed not set
        // kmeans++
        kmeans_init_centers3(center.s, s,index2,swarms,324324);
        run_kmeans3(s_group,s,center.s,500,index2,swarms);
    }

    // Account for translation to origin of coordinates
    #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<swarms; i++){
        double *current_s = &center.s[i*3];
        for(int j = 0; j<3; j++){
            current_s[j] += receptor.translation[j];
        }
    }
    // Create points over the surface of each surface cluster

    delete [] surface_centroids;
    delete [] group;
    delete [] surface_coord;
    delete [] sampling;
    delete [] sample_filterd;
    delete [] indices;
    delete [] s;
    delete [] s_group;
}


void calculate_start_pos(
    SwarmCenters & center,
    Complex & receptor, Complex & ligand, int swarms,int glowworms, int start_points_seed, 
    double * rec_translation,  double * lig_translation,
    double surface_density, bool use_anm, int anm_seed,
    int anm_rec, int anm_lig, bool membrane, bool transmambrane,
    bool write_starting_positions, double swarm_radius, bool flip,
    double fixed_distance, int swarms_per_restraint, bool dense_smapling, const char * rec_name, int rank, int size
){
    // int num_threads = get_env_num_threads();
    if(rank == 0){
        std::cout<<"  Calculating starting positions... "<<std::endl;
        std::cout<<"  * Surface density: TotalSASA/"<<surface_density<<std::endl;
        std::cout<<"  * Swarm radius: "<<swarm_radius<<"Å"<<std::endl;
        std::cout<<"  * 180° flip of 50% of starting poses:"<<flip<<std::endl;
    }
    
    // random number generator
    int number_of_points = DEFAULT_SPHERES_PER_CENTROID;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(start_points_seed + rank); //Standard mersenne_twister_engine seeded with rd()
    // std::mt19937 nm_gen(anm_seed); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    // double generator_value = distribution(gen);
    std::default_random_engine nm_gen(anm_seed);
    std::normal_distribution<double> distribution_nm(DEFAULT_EXTENT_MU, DEFAULT_EXTENT_SIGMA);
    // unsigned int seeds = 324324;
    // distribution_nm.seed(anm_seed);
    bool has_membrane = membrane || transmambrane;

    // TODO: restrains

    // receptor membrane
    if(has_membrane){
        std::cerr<<"developing"<<std::endl;
    }

    // calculate surface_point
    /* define the MPI chunks */
    calculate_surface_point(
    center,
    receptor, ligand, swarms, glowworms,  start_points_seed, 
    rec_translation,  lig_translation,
     surface_density,  use_anm,  anm_seed,
     anm_rec,  anm_lig,  membrane,  transmambrane,
     write_starting_positions,  swarm_radius,  flip,
     fixed_distance,  swarms_per_restraint,  dense_smapling,  rec_name,  number_of_points, rank, size
    );
    /* each proc executes all swarms */
    // print_mat(center.s, center.swarms, 3);
    // std::cout<<"glowworms"<<glowworms<<std::endl;
    // TODO: each process holds its own pos, would result in scalability
    center.pos = new double [center.swarms * glowworms*(7+anm_rec+anm_lig)];
    center.pos_len = glowworms*(7+anm_rec+anm_lig);
    center.anm_rec = anm_rec;
    center.anm_lig = anm_lig;
    // print_mat(center.s,center.swarms,3);
    // #pragma omp parallel for num_threads(num_threads)
    for(int i = 0; i<center.swarms;i++){
        // populate poses Creates new poses around a given center and a given radius
        double * poses = &center.pos[i*glowworms*(7+anm_rec+anm_lig)];
        double * swarm_center = &center.s[i*3];
        // std::cout<<glowworms<<std::endl;
        for(int j = 0; j<glowworms; j++){

            // First calculate a random translation within the swarm sphere
            double * center_position = &poses[j*(7+anm_rec+anm_lig)];

            double generate_pos[3];
            double r2 = swarm_radius * swarm_radius;
            while (true)
            {
                double generate_value0 = distribution(gen);
                double generate_value1 = distribution(gen);
                double generate_value2 = distribution(gen);
                double x = (2 * generate_value0 - 1) * swarm_radius;
                double y = (2 * generate_value1 - 1) * swarm_radius;
                double z = (2 * generate_value2 - 1) * swarm_radius;
                // std::cout<<"value:"<<generate_value0<<generate_value1<<generate_value2<<std::endl;
                if(x * x + y * y + z * z <= r2) {
                    generate_pos[0] = x;
                    generate_pos[1] = y;
                    generate_pos[2] = z;
                    break;
                }
            }
            // print_array(generate_pos,3);

            // get_random_point_within_sphere(generate_pos,swarm_radius,generator_value);
            center_position[0] = swarm_center[0] + generate_pos[0];
            center_position[1] = swarm_center[1] + generate_pos[1];
            center_position[2] = swarm_center[2] + generate_pos[2];
            // no restraints
            double u1 = distribution(gen);
            double u2 = distribution(gen);
            double u3 = distribution(gen);
            // std::cout<<u1<<" "<<u2<<" "<<u3<<std::endl;
            center_position[3] = sqrt(1-u1)*sin(2*PI*u2);
            center_position[4] = sqrt(1-u1)*cos(2*PI*u2);
            center_position[5] = sqrt(u1)*sin(2*PI*u3);
            center_position[6] = sqrt(u1)*cos(2*PI*u3);
            // print_array(&center_position[3],4);

            // create pose given the center
            if (use_anm){
                if(anm_rec>0){
                    for(int index = 0; index<anm_rec; index++){
                        double nm_value = distribution_nm(nm_gen);
                        center_position[index+7] = nm_value;
                    }
                }
                if(anm_lig>0){
                    for(int index = 0; index<anm_lig; index++){
                        double nm_value = distribution_nm(nm_gen);
                        center_position[index+7+anm_rec] = nm_value;
                    }
                }
            }

        }
        
    }

    
  
}


void write_center(double * center, std::ofstream &newFile, int swarms){
    if (newFile.is_open()) {         
        for(int nn = 0; nn<swarms; nn++){
            string type = "ATOM";
            // type = string(6 - type.length(), ' ') + type;
            string number = to_string(nn+1);
            number = string(7 - number.length(), ' ') + number;
            string atom_name = "H";
            if(atom_name.length()<4){
                atom_name = " "+atom_name;
            }
            // std::cout<<receptor.atoms[nn].alternative<<1212<<std::endl;
            atom_name = atom_name+string(4 - atom_name.length(), ' ');
            string alternative = "";

            string residue_name = "SWR";
            residue_name = string(4 - residue_name.length(),' ') + residue_name ;
            string chain_id = "Z";
            chain_id = string(2 - chain_id.length(), ' ')+chain_id;

            string residue_number = to_string(nn+1);
            residue_number = string(4 - residue_number.length(),' ') + residue_number ;
            string residue_insertion = " ";
            residue_insertion = string(1 - residue_insertion.length(),' ') + residue_insertion ;

            string blank = "   ";

            double xx = center[nn*3];
            double yy = center[nn*3+1];
            double zz = center[nn*3+2];
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

            double occupancy = 1;
            std::stringstream s_o;
            s_o << std::fixed << std::setprecision(2) << occupancy;
            string oo = s_o.str();
            oo = string(6 - oo.length(),' ') + oo ;

            double b_factor = 1;
            std::stringstream s_b;
            s_b << std::fixed << std::setprecision(2) << b_factor;
            string bb = s_b.str();
            bb = string(6 - bb.length(),' ') + bb ;

            string element = "H";
            element = string(12 - element.length(),' ') + element ;
            // TODO: 重复
            
            // zz_str = string(8 - zz_str.length(),' ') + zz_str ;

            // string residue_name = receptor.atoms[nn].residue_name;
            newFile << type + number + " "+atom_name+ alternative+ residue_name+chain_id
                + residue_number+residue_insertion+blank+xx_str+yy_str+zz_str+oo+bb+element+"\n";
        }
        // exit(-1);      
    // 写入文件内容
    // newFile << "#Coordinates  RecID  LigID  Luciferin  Neighbor's number  Vision Range  Scoring\n";
    }
}


void points_on_sphere(double *sphere_points, int number_of_points){
    double increment = PI * (3.0-sqrt(5.0));
    double offset = 2.0 / number_of_points;
    for (int i = 0; i< number_of_points; i++){
        double y = i * offset - 1.0 + (offset / 2.0);
        double r = sqrt(1-y*y);
        double phi = i * increment;
        sphere_points[i*3] = cos(phi)*r;
        sphere_points[i*3+1] = y;
        sphere_points[i*3+2] = sin(phi)*r;

    }
}

int select_surface(double *surface, Complex & complex){
    int size=0;
    for(int i = 0; i<complex.num_res_type;i++){
        // if surface
        string res_name = complex.residue_name[i];
        int iter = complex.residue_atom_number[i+1]-complex.residue_atom_number[i];
        bool is_surface = false;
        for(int j = 0; j<26;j++){
            if (res_name == complex.surface[j])
            {
                is_surface = true;
                // std::cout<<res_name<<" "<<iter<<std::endl;
                break;
            }
            
            
        }
        if(is_surface){
            int rec_start = complex.residue_atom_number[i];
            int rec_end = complex.residue_atom_number[i+1];
            int len = rec_end - rec_start;

            memcpy(surface+size*3,complex.atom_coordinates+rec_start*3,len*3*sizeof(double));

            size+= len;
        }
        
    }
    return size;
    // print_mat(surface,)
    // std::cout<<"size:"<<size<<std::endl;
}

void get_random_point_within_sphere(double * pos, double radius, double number_generator){
    double r2 = radius * radius;
    while (true)
    {
        double x = (2 * number_generator - 1) * radius;
        double y = (2 * number_generator - 1) * radius;
        double z = (2 * number_generator - 1) * radius;
        if(x * x + y * y + z * z <= r2) {
            pos[0] = x;
            pos[1] = y;
            pos[2] = z;
            break;
        }
    }
    
}



void cal_dist(double * distances_matrix_rec, double *atom_coordinates, int num_atoms, int dis_size){
    int index = 0;
    for(int i = 0; i<num_atoms; i++){
        for(int j = i+1; j < num_atoms; j++){

            double distance = 0.0;
            for(int d = 0; d<3;d++){
                double diff =  atom_coordinates[i*3+d] - atom_coordinates[j*3+d];
                distance += diff * diff;
            }
            distances_matrix_rec[index++] = sqrt(distance);
        }
    }
}



