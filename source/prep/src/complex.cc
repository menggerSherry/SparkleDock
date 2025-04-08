#include <iostream>
#include "complex.h"
#include "lib_tools.h"
using namespace std;
void generate_complex(
    Complex & complex, vector<vector<Atom>> atom_list, vector<vector<Chain>> chain_list, vector<vector<Residue>> residue_list,
    vector<string> files
){
    // TODO: only represent elements
    complex.atoms = atom_list[0];
    complex.chain = chain_list[0];
    complex.residue = residue_list[0];
    vector<double> coord;
    vector<int> atom_index;
    vector<int> residue_index;
    // index
    for(int i = 0; i< complex.atoms.size();i++){
        atom_index.push_back(i);
    }
    for(int i = 0; i< complex.residue.size();i++){
        residue_index.push_back(i);
    }
    complex.atom_index = new int[complex.atoms.size()];
    complex.residue_index = new int[complex.residue.size()];
    memcpy(complex.atom_index,&atom_index[0],sizeof(int)*complex.atoms.size());
    memcpy(complex.residue_index, &residue_index[0],sizeof(int)*complex.residue.size());

    // num structures and number atoms
    complex.num_structures = atom_list.size();
    complex.num_atoms = complex.atoms.size();
    
    // convert coord to SoA
    // vector<int > structure_size(complex.num_structures);
    complex.structure_size = new int[complex.num_structures+1];
   
    memset(complex.structure_size, 0, sizeof(int)*(complex.num_structures+1));
    
    for(int i = 0; i<atom_list.size();i++){
        complex.structure_size[i+1] = atom_list[i].size();
        
        // vector<Atom> atoms = atom_list[i];
        for(int j = 0; j<atom_list[i].size();j++){
            coord.push_back(atom_list[i][j].x);
            coord.push_back(atom_list[i][j].y);
            coord.push_back(atom_list[i][j].z);
        }
    }

    for(int i = 0; i<complex.num_structures; i++){
        complex.structure_size[i+1]+=complex.structure_size[i];
    }
    // for(int i = 0; i<complex.num_structures+1; i++){
    //     std::cout<<" "<<complex.structure_size[i];
    // }
    // std::cout<<"11111:"<<complex.structure_size[0]<<std::endl;
    complex.coord_size = coord.size();
    
    complex.atom_coordinates = new double[coord.size()];
    memcpy(complex.atom_coordinates,&coord[0],coord.size()*sizeof(double));
    complex.representative_id = 0;

    // construct residue/////////////////////////////////////////////////
    complex.residue_size = new int[complex.num_structures+1];
    memset(complex.residue_size,0,sizeof(int)*(complex.num_structures+1));
    vector <int> atom_size;
    complex.residue_id = new int[residue_list[0].size()];
    // 几个蛋白质文件
    for(int i = 0; i<residue_list.size();i++){
        complex.residue_size[i+1] = residue_list[i].size();
        // 遍历残差
        for(int j = 0; j<residue_list[i].size(); j++){
            
            complex.residue_name.push_back(residue_list[i][j].name);
            complex.residue_id[j] = residue_list[i][j].number;
            // complex.residue_id.push_back(residue_list[i][j].number);
            atom_size.push_back(residue_list[i][j].atom_number);
        }
    }
    for(int i = 0; i<complex.num_structures; i++){
        complex.residue_size[i+1]+=complex.residue_size[i];
    }

    complex.num_res_type = atom_size.size();
    complex.residue_atom_number = new int[atom_size.size()+1];

    memset(complex.residue_atom_number,0,sizeof(int)*atom_size.size()+1);
    for(int i = 0;i<atom_size.size();i++){
        complex.residue_atom_number[i+1] = complex.residue_atom_number[i]+atom_size[i];
    }

    
    for(int i = 0; i<atom_list.size();i++){
        // vector<Atom> atoms = atom_list[i];
        
        for(int j = 0; j<atom_list[i].size();j++){
            
            complex.elements.push_back(atom_list[i][j].element);
            complex.atom_name.push_back(atom_list[i][j].name);

        }
    }
    complex.elements_size = complex.elements.size();
    // complex.elements = new string[complex.elements_size];

    // memcpy(complex.elements,&elements[0],complex.elements_size*sizeof(char));

    // atom mass map
    complex.mass = new double[complex.elements_size];
    for(int i = 0 ; i<complex.elements_size; i++){
        string elem = complex.elements[i];

        map<string,double>::iterator it = complex.MASSES.find(elem);
        if(it != complex.MASSES.end()){
            complex.mass[i] = it->second;
        }
        else{
            std::cout<<"error type:"<<elem<<std::endl;
        }

    }

    // init

    complex.translation = new double [3];

    // mask
    complex.mask = new int [complex.elements_size];
    memset(complex.mask, 0 , sizeof(int)*complex.elements_size);
    // std::cout<<complex.elements_size<<endl;


}

void free_complex(Complex & complex){
    delete [] complex.atom_index;
    delete [] complex.residue_index;
    delete [] complex.atom_coordinates;
    delete [] complex.structure_size;
    delete [] complex.mass;
    delete [] complex.translation;
    delete [] complex.residue_atom_number;
    delete [] complex.residue_size;
    delete [] complex.mask;
    delete [] complex.residue_id;
    delete [] complex.modes;
}


void get_nm_mask(
    int * mask,
    vector<string> residue_names,
    map<string,string> STANDARD_TYPES,
    map<string, string> MODIFIED_TYPES,
    string * DNA_STANDARD_TYPES,
    string * RNA_STANDARD_TYPES,
    int * residue_atom_number,
    int * residue_size
    
){
    // TODO: single;
    int num_threads = get_env_num_threads();
    int residue_start = residue_size[0];
    int residue_end = residue_size[1];
    #pragma omp parallel for num_threads(num_threads)
    for(int i = residue_start; i<residue_end;i++){
        int mask_start = residue_atom_number[i];
        int mask_end = residue_atom_number[i+1];
        // std::cout<<mask_end - mask_start<<std::endl;
        string residue_name  = residue_names[i];
        bool is_protein=false;
        bool is_nucleic = false;
        // protein
        map<string,string>::iterator it_protein_standard = STANDARD_TYPES.find(residue_name);
        map<string, string>::iterator it_protein_modified = MODIFIED_TYPES.find(residue_name);
        if(it_protein_standard != STANDARD_TYPES.end() || it_protein_modified != MODIFIED_TYPES.end()){
            is_protein = true;
        }

        for(int idx = 0; idx<5;idx++){
            if(residue_name == DNA_STANDARD_TYPES[idx]){
                
                is_nucleic = true;
                break;
            }
        }
        for(int idx = 0; idx< 5; idx++){
            if(residue_name == RNA_STANDARD_TYPES[idx]){
                
                is_nucleic = true;
                break;
            }
        }
        if(is_protein || is_nucleic){
            
            for(int j = mask_start;j<mask_end;j++){
                mask[j] = 1;
            }
        }else{
            
            for(int j = mask_start;j<mask_end;j++){
                mask[j] = 0;
            }
        }
        // std::cout<<"count"<<count<<std::endl;
        // std::cout<<complex.residue.atom_number<<std::endl;
    }
    

}
