from Bio.PDB import PDBParser, Superimposer, NeighborSearch
import numpy as np
from prody import parsePDB, ANM, extendModel, confProDy
import prody
import pandas as pd
import sys
from scipy.spatial.distance import cdist

# CAPRI criteria

def cal_rmsd_and_fnat(ref_pdb, pred_pdb,cutoff_distance=8):
    ref_molecule = parsePDB(str(ref_pdb))
    pred_molecule = parsePDB(str(pred_pdb))
    
    
    backbone_ref_rec = ref_molecule.select("name CA")
    if not backbone_ref_rec:
        # If previous step has failed, maybe we're dealing with DNA
        backbone_ref_rec = ref_molecule.select("name C4")
    
    backbone_pred_rec = pred_molecule.select("name CA")
    if not backbone_pred_rec:
        # If previous step has failed, maybe we're dealing with DNA
        backbone_pred_rec = pred_molecule.select("name C4")
        
    transformation = prody.calcTransformation(backbone_ref_rec, backbone_pred_rec)
    structure1_superposed_ref = transformation.apply(ref_molecule)
        
        
    backbone_ref = structure1_superposed_ref.select("chain B and name CA")
    if not backbone_ref:
        # If previous step has failed, maybe we're dealing with DNA
        backbone_ref = structure1_superposed_ref.select("chain B and name C4")
    
    backbone_pred = pred_molecule.select("chain B and name CA")
    if not backbone_pred:
        # If previous step has failed, maybe we're dealing with DNA
        backbone_pred = pred_molecule.select("chain B and name C4")
        
    rmsd = prody.calcRMSD(backbone_ref, backbone_pred)
    
    # fnat
    # Calculate native contacts in the reference structure
    native_contacts = prody.buildDistMatrix(backbone_ref_rec) < cutoff_distance

    # Calculate native contact count
    native_contact_count = native_contacts.sum()

    # Calculate contacts in the predicted structure
    predicted_contacts = prody.buildDistMatrix(backbone_pred_rec) < cutoff_distance

    # Calculate shared contacts (intersection of native and predicted contacts)
    shared_contacts = np.logical_and(native_contacts, predicted_contacts)

    # Calculate FNAT
    fnat = shared_contacts.sum() / native_contact_count
    
    # irmsd
    lig_coord_ref = backbone_ref.getCoords()
    lig_coord_pred = backbone_pred.getCoords()
    
    rec_ref_bacbone = ref_molecule.select("chain A and name CA")
    if not backbone_ref_rec:
        # If previous step has failed, maybe we're dealing with DNA
        rec_ref_bacbone = ref_molecule.select("chain A and name C4")
        
    rec_coord_ref = rec_ref_bacbone.getCoords()
    distance = cdist(lig_coord_ref,rec_coord_ref)
    indices = np.where(distance < 10)
    interface_lig_ref = lig_coord_ref[indices[0]]
    interface_lig_pred = lig_coord_pred[indices[0]]
    i_rmsd = np.sqrt(np.mean((interface_lig_ref - interface_lig_pred)**2))
    # print(i_rmsd)
    # print(interface_lig_pred.shape)
    return rmsd, fnat, i_rmsd
        

arguments = sys.argv


scores = pd.read_csv("rank_by_scoring.list",sep = "\t")

# print(scores.head(5))
pred_files = []

for row in scores.index:
    line = scores.iloc[row,0].split()
    swarm = int(line[0])
    glowworm_id = int(line[1])
    files = "./swarm_%d/lightdock_%d.pdb"%(swarm,glowworm_id)
    pred_files.append(files)
   
# for score in scores:
#     # print(score)
#     swarm = score[0]
#     print(swarm)
#     glowworm_id = score[1]
#     files = "./swarm_%d/lightdock_%d.pdb"%(swarm,glowworm_id)
#     pred_files.append(files)

# print(pred_files)


rmsds = []
fnats = []
i_rmsds = []
reference = "1A2K_segid.pdb"
result = pd.DataFrame(columns=['file', 'l-rmsd','i-rmsd', 'fnat', 'criteria'])

for file in pred_files:
    rmsd,fnat,i_rmsd = cal_rmsd_and_fnat(reference, file)
    rmsds.append(rmsd)
    fnats.append(fnat)
    i_rmsds.append(i_rmsd)
    criteria = ''
    if fnat < 0.1 or rmsd > 10.0:
        criteria = 'incorrect'
    elif (fnat>=0.1) and ((rmsd>5.0 and rmsd<=10) or (i_rmsd>2.0 and i_rmsd<=4.0)):
        criteria = 'acceptable'
    elif (fnat>=0.3) and ((rmsd>1 and rmsd <=5) or (i_rmsd>1 and i_rmsd<=2.0)):
        criteria = 'medium'
    elif (fnat>=0.5) and ((rmsd<=1.0) or (i_rmsd<=1.0)):
        criteria = 'high'
    else:
        criteria = 'in'
    datarow = {'file':file, 'l-rmsd':rmsd, 'i-rmsd': i_rmsd, 'fnat':fnat, 'criteria':criteria}
    result = result.append(datarow, ignore_index=True)
    
# pred = "./swarm_20/lightdock_156.pdb"

result.to_csv('criteria.csv', index=False)




# rmsd = np.sqrt(((ref_coords - pred_coords) ** 2).mean())

# print(i_rmsds)