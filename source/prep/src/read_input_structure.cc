#include "read_input_structure.h"
#include "json/json.h"
// using namespace std;
#include "config_args.h"

void get_setup(ConfigArgs & para, string file_){
    Json::Reader reader;
    Json::Value root;

    ifstream in(file_,ios::binary);

    if(!in.is_open()){
        cout<<"Error opening"<<endl;
        return;
    }

    if(reader.parse(in,root)){
        para.anm_lig = root["anm_lig"].asInt();
        para.anm_lig_rmsd = root["anm_lig_rmsd"].asDouble();
        para.anm_rec = root["anm_rec"].asInt();
        para.anm_rec_rmsd = root["anm_rec_rmsd"].asDouble();
        para.anm_seed = root["anm_seed"].asInt();
        para.dense_sampling = root["dense_sampling"].asBool();
        para.fixed_distance = root["fixed_distance"].asDouble();
        para.flip = root["flip"].asBool();
        para.glowworms = root["glowworms"].asInt();
        para.ligand_pdb = root["ligand_pdb"].asString();
        para.membrane  = root["membrane"].asBool();
        para.noh = root["noh"].asBool();
        para.now = root["now"].asBool();
        para.noxt = root["noxt"].asBool();
        para.receptor_pdb = root["receptor_pdb"].asString();
        para.starting_points_seed = root["starting_points_seed"].asInt();
        para.surface_density = root["surface_density"].asDouble();
        para.swarm_radius = root["swarm_radius"].asDouble();
        //  para.swarms = root["swarms"].asInt();
        para.swarms = root["swarms"].asInt();
        //  TODO: swarm is calculated 
        para.swarms_per_restraint = root["swarms_per_restraint"].asInt();
        para.transmembrane = root["transmembrane"].asBool();
        para.use_anm = root["use_anm"].asBool();
        para.verbose_parser = root["verbose_parser"].asBool();
        para.write_starting_positions = root["write_starting_positions"].asBool();
        para.translation_step = 0.5;
        para.rotation_step = 0.5;
        para.nmodes_step = 0.5;
        para.local_minimization = false;
        
        para.rho = root["rho"].asDouble();
        para.gamma = root["gamma"].asDouble();
        para.vision_range = root["initialVisionRange"].asDouble();
        para.beta = root["beta"].asDouble();
        para.luciferin = root["initialLuciferin"].asDouble();
        para.max_neighbor = root["maximumNeighbors"].asInt();
        para.max_vision_range = root["maximumVisionRange"].asDouble();
    }

    in.close();
}




void read_input_structure(
    Complex & complex,
    const char * pdb_file_name,
    bool ignore_oxt,
    bool ignore_hydrogons,
    bool ignore_water,
    bool verbose,
    int N
){

    vector<string> residues_to_ignore;
    vector<string> atoms_to_ignore;
    
    char * f = new char[N+1];
    strcpy(f, pdb_file_name);
    char * p = strtok(f, " ");
    std::vector<std::string> files;
    while(p){
        files.push_back(p);
        p = strtok(NULL, " ");
    }
    int size = files.size(); // 文件size

    if(ignore_oxt)
    {
        atoms_to_ignore.push_back("OXT");
        // std::cout<<""<<std::endl;
    }
    if(ignore_hydrogons)
    {
        atoms_to_ignore.push_back("H");

    }
    if(ignore_water){
        residues_to_ignore.push_back("HOH");
    }
    vector<vector<Atom>> atom_list;
    vector<vector<Chain>> chain_list;
    vector<vector<Residue>> residue_list;
    // std::cout<<N<<std::endl;
    vector<string> files_list;
    for(int i = 0; i< size; i++){
        vector<Atom> atoms;
        vector<Chain> chains;
        vector<Residue> residues;
        

        std::string cur_file = files[i];
        parse_complex_from_file(atoms,chains, residues, cur_file,residues_to_ignore,atoms_to_ignore);
        atom_list.push_back(atoms);
        chain_list.push_back(chains);
        residue_list.push_back(residues);
        files_list.push_back(cur_file);
        // std::cout<<atoms.size()<<" atoms, and "<<residues.size()<<" residues read"<<std::endl;

    }
    // consturct complex
    
    generate_complex(complex,atom_list,chain_list,residue_list,files_list);

    
    delete [] f;
    

}


Atom read_atom_line(string line, string line_type, vector<string> residues_to_ignore, vector<string> atoms_to_ignore){
    // read atoms
    // TODO: string element = line.substr(76,2);
    Atom atom;
    double x = stod(line.substr(30,8));
    double y = stod(line.substr(38,8));
    double z = stod(line.substr(46,8));

    // std::cout<<x<<","<<y<<","<<z<<std::endl;
    int atom_number =  stoi(line.substr(6,5));

    string atom_name = line.substr(12,4);
    trim(atom_name);
    // std::cout<<atom_name<<std::endl;
    string atom_alternative = line.substr(16,1);
    trim(atom_alternative);
    string residue_name = line.substr(17,4);
    trim(residue_name);
    string chain_id = line.substr(21,1);
    trim(chain_id);
    string residue_insertion = line.substr(26,1);
    int residue_number = stoi(line.substr(22,4));

    // resdue to ignore
    for(int i = 0; i<residues_to_ignore.size(); i++){
        if(residue_name == residues_to_ignore[i]){
            // std::cout<<"ignore atom"<<std::endl;
            atom.valid = false;
        }
    }
    // atom to ignore
    bool H = false;
    bool atom_in = false;
    for(int i =0; i<atoms_to_ignore.size();i++){
        if("H" == atoms_to_ignore[i])
        {
            H = true;
            
        }
        if(atom_name == atoms_to_ignore[i]){
            atom_in = true;
        }
    }
    if((H && atom_name[0]=='H') || atom_in){
        // std::cout<<"ignore atom"<<std::endl;
        atom.valid = false;
    }
    double occupancy = 1;
    try{
        size_t idx;
        string s = line.substr(54,6);
            occupancy = stof(s,&idx);
            if(idx<s.size()) occupancy = 1.0;
    }catch(...){
        occupancy = 1.0;
    }

    double b_factor = 0;
    try{
        size_t idx;
        string s = line.substr(60,6);

        b_factor = stof(s,&idx);
        if(idx<s.size()) b_factor = 0;
    }catch(...){
        b_factor = 0.0;
    }
    atom.type = line.substr(0,6);
    atom.number = atom_number;
    atom.name = atom_name;
    atom.alternative = atom_alternative;
    atom.chain_id = chain_id;
    atom.residue_name = residue_name;
    atom.residue_number = residue_number;
    atom.residue_insertion = residue_insertion;
    atom.x = x;
    atom.y = y;
    atom.z = z;
    atom.occupancy = occupancy;
    atom.b_factor = b_factor;
    // assign element
    atom.element = atom.name.substr(0,2);
    bool inside = false;
    for(int i = 0; i<12; i++){
        if(atom.element == atom.recgnnized_elements[i]){
            inside = true;
        }
    }
    if(!inside){
        atom.element = atom.name.substr(0,1);
        bool inside2 = false;
        for(int i = 0; i<12; i++){
            if(atom.element == atom.recgnnized_elements[i]){
                inside2 = true;
            }
        }
        if(!inside2){
            std::cout<<"error not inside"<<std::endl;
            exit(0);
        }
    }
    // atom.element = line.substr(76,2);
    trim(atom.element);
    // std::cout<<"<"<<atom.element<<">"<<std::endl;
    return atom;
    
}


void parse_complex_from_file(
   vector<Atom> & atoms, vector <Chain> & chains, vector <Residue> & residues, string path, vector<string> residues_to_ignore, vector<string> atoms_to_ignore
){
    
    // std::string path = "example.txt";
    std::ifstream ifs(path);
    
    
    string last_chain_id = "#";
    string last_residue_name = "#";
    int last_residue_number = NULL;
    string last_residue_insertion = "#";
    if (!ifs.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        exit(0);
    }

    // Read the file content line by line and output to console
    std::string line;
    int count = 0;
    int num_models = 0;
    while (std::getline(ifs, line)) {
        if(num_models <=1){
            string line_type = line.substr(0,6);
            trim(line_type);
            if(line_type == "MODEL"){
                num_models += 1;
                if(num_models > 1){
                    std::cout<<"Multiple models found in"<<path<<". Only first model will be used"<<std::endl;
                }
            }
            else if(line_type == "ATOM" || line_type == "HETATM"){
                
                Atom atom = read_atom_line(line,line_type,residues_to_ignore,atoms_to_ignore);
                if(atom.valid) atoms.push_back(atom);
                else continue;
                if(last_chain_id != atom.chain_id){
                    last_chain_id = atom.chain_id;
                    Chain current_chain;
                    current_chain.cid = last_chain_id;
                    chains.push_back(current_chain);
                }
                if(last_residue_name != atom.residue_name || last_residue_number != atom.residue_number || last_residue_insertion != atom.residue_insertion){
                    //TODO: under modified
                    last_residue_name = atom.residue_name;
                    last_residue_number = atom.residue_number;
                    last_residue_insertion = atom.residue_insertion;
                    Residue current_residue;
                    current_residue.name = atom.residue_name;
                    current_residue.number = atom.residue_number;
                    current_residue.insertion = atom.residue_insertion;
                    residues.push_back(current_residue);
                    // std::cout<<current_residue.name<<std::endl;
                }
                residues.back().atom_number++;

            }
            
            // Trim(line_type);
            // std::cout<<"<"<< line_type<<">"<<std::endl;
        }
        // std::cout << line << std::endl;
        count++;
        // std::cout<<count<<std::endl;
    }
    
    ifs.close();
    // std::cout<<count<<std::endl;
}